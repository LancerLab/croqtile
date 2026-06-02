#ifndef __CHOREO_FRAGMENT_LAYOUT_HPP__
#define __CHOREO_FRAGMENT_LAYOUT_HPP__

#include <algorithm>
#include <cstddef>
#include <string>

namespace Choreo {

// Target-agnostic parameters for a dim-1 (row-wise) fragment reduction.
// Computed once by FragmentLayout::GetReduceParams() and consumed by any
// target's codegen without the codegen needing to know the layout kind.
struct ReduceParams {
  size_t rows_per_thread = 0; // outer loop bound
  size_t local_cols = 0;      // inner loop bound (columns per row per thread)
  size_t threads_per_row = 1; // AllReduce width
  size_t thread_count = 0;    // total threads (for workspace sizing)

  bool NeedsWorkspace() const { return threads_per_row > 32; }
};

enum class LayoutKind {
  UNIFORM,
  WGMMA_ACC,
  WGMMA_RS_A,
  REPLICATED_1D,
  CTMMA_ACC,
};

inline const std::string STR(LayoutKind k) {
  switch (k) {
  case LayoutKind::UNIFORM: return "UNIFORM";
  case LayoutKind::WGMMA_ACC: return "WGMMA_ACC";
  case LayoutKind::WGMMA_RS_A: return "WGMMA_RS_A";
  case LayoutKind::REPLICATED_1D: return "REPLICATED_1D";
  case LayoutKind::CTMMA_ACC: return "CTMMA_ACC";
  }
  return "?";
}

struct FragmentLayout {
  LayoutKind kind = LayoutKind::UNIFORM;

  size_t regs_per_thread = 0;
  size_t thread_count = 0;
  std::string thread_count_expr;

  size_t logical_rows = 0;
  size_t logical_cols = 0;

  // Vec4 grouping width for coalesced memory access. When vec_width > 1,
  // consecutive `vec_width` registers per thread map to consecutive memory
  // addresses, enabling vectorized float4 loads/stores. Modeled after
  // TileLang's register layout.
  size_t vec_width = 1;

  // WGMMA-specific
  size_t warp_count = 0;
  size_t rows_per_thread = 0;
  size_t cols_per_group = 0;
  size_t threads_per_row = 0;

  // REPLICATED_1D-specific
  size_t replicate = 0;
  // For UNIFORM-derived REPLICATED_1D: row stride for LogicalRowFromReg.
  // When nonzero, row = r * reduce_row_stride + tid / replicate.
  // When zero (WGMMA-derived), use the WGMMA warp/lane formula.
  size_t reduce_row_stride = 0;

  // Source tracking
  std::string anchor_symbol;

  bool IsVectorizable() const { return vec_width >= 4; }

  // ---------------------------------------------------------------------------
  // ForwardIndex: logical (i, j) -> physical register index within a thread.
  //
  // Derived by inverting the hardware store pattern. For WGMMA_ACC, the store
  // function (Policy_WGMMA_D_M64K16::store) maps d[reg] -> D(row, col):
  //
  //   D(row0, col0) = d[c*4+0]     where c = col_group index
  //   D(row0, col1) = d[c*4+1]           col0 = c*8 + (tid%4)*2
  //   D(row1, col0) = d[c*4+2]           col1 = col0 + 1
  //   D(row1, col1) = d[c*4+3]           row0 = warp*16 + lane/4
  //                                       row1 = row0 + 8
  //
  // Inverting: given logical (i, j), the register index is:
  //   c = j / 8   (which group of 8 columns)
  //   row_offset = (i % 16) / 8   (0 for row0, 1 for row1)
  //   col_parity = j % 2          (0 for col0, 1 for col1)
  //   reg = c * 4 + row_offset * 2 + col_parity
  //
  // Cross-validated against TileLang's forward_index formula.
  // ---------------------------------------------------------------------------
  std::string ForwardIndex(const std::string& i,
                           const std::string& j = "") const {
    switch (kind) {
    case LayoutKind::WGMMA_ACC:
      return "((" + j + ") / 8 * 4 + (" + i + ") % 16 / 8 * 2 + (" + j +
             ") % 2)";
    case LayoutKind::REPLICATED_1D: return "((" + i + ") % 16 / 8)";
    case LayoutKind::CTMMA_ACC:
      return "((" + i + ") * " + std::to_string(logical_cols) + " + (" + j +
             ")) / " + std::to_string(thread_count) + ")";
    case LayoutKind::WGMMA_RS_A:
      return "((" + j + ") / 8 * 4 + (" + i + ") % 16 / 8 * 2 + (" + j +
             ") % 2)";
    case LayoutKind::UNIFORM:
      if (vec_width > 1) {
        std::string W = std::to_string(vec_width);
        std::string TW = std::to_string(thread_count * vec_width);
        std::string flat =
            j.empty() ? ("(" + i + ")")
                      : ("(" + i + " * " + std::to_string(logical_cols) +
                         " + " + j + ")");
        return "(" + flat + " / " + TW + " * " + W + " + " + flat + " % " + W +
               ")";
      }
      if (j.empty()) return "((" + i + ") / " + thread_count_expr + ")";
      return "((" + i + " * " + std::to_string(logical_cols) + " + " + j +
             ") / " + thread_count_expr + ")";
    }
    return "0";
  }

  // ---------------------------------------------------------------------------
  // ForwardThread: logical (i, j) -> threadIdx that owns the element.
  //
  // For WGMMA_ACC, derived from the same store pattern:
  //   warp = i / 16
  //   lane = (i % 8) * 4 + (j % 8) / 2
  //   tid = warp * 32 + lane
  // ---------------------------------------------------------------------------
  std::string ForwardThread(const std::string& i,
                            const std::string& j = "") const {
    switch (kind) {
    case LayoutKind::WGMMA_ACC:
      return "((" + i + ") / 16 * 32 + (" + i + ") % 8 * 4 + (" + j +
             ") % 8 / 2)";
    case LayoutKind::REPLICATED_1D:
      return "((" + i + ") / 16 * 32 + (" + i + ") % 8 * 4)";
    case LayoutKind::UNIFORM:
      if (vec_width > 1) {
        std::string W = std::to_string(vec_width);
        std::string flat =
            j.empty() ? ("(" + i + ")")
                      : ("(" + i + " * " + std::to_string(logical_cols) +
                         " + " + j + ")");
        return "(" + flat + " / " + W + " % " + thread_count_expr + ")";
      }
      if (j.empty()) return "((" + i + ") % " + thread_count_expr + ")";
      return "((" + i + " * " + std::to_string(logical_cols) + " + " + j +
             ") % " + thread_count_expr + ")";
    default: return "0";
    }
  }

  // Row index extracted from physical register index (for cross-fragment
  // broadcast in automap). Maps register -> which-row-within-thread.
  // WGMMA_ACC: each thread owns 2 rows; (reg & 3) >> 1 gives 0 or 1.
  std::string RowFromRegIndex(const std::string& r) const {
    switch (kind) {
    case LayoutKind::WGMMA_ACC:
    case LayoutKind::WGMMA_RS_A: return "(((" + r + ") & 3) >> 1)";
    case LayoutKind::REPLICATED_1D: return "(" + r + ")";
    default: return "0";
    }
  }

  // -------------------------------------------------------------------------
  // Reduction support
  // -------------------------------------------------------------------------

  // Compute target-agnostic parameters for a dim-1 (row-wise) reduction.
  // Called during layout pass (MakeReduceDst) and by codegen.
  ReduceParams GetReduceParams() const {
    ReduceParams rp;
    rp.thread_count = thread_count;
    if (kind == LayoutKind::WGMMA_ACC) {
      rp.rows_per_thread = rows_per_thread;
      rp.local_cols =
          rows_per_thread > 0 ? regs_per_thread / rows_per_thread : 0;
      rp.threads_per_row = threads_per_row;
    } else if (kind == LayoutKind::UNIFORM && logical_cols > 1 &&
               thread_count > 0) {
      rp.local_cols = std::max((size_t)1, logical_cols < thread_count
                                              ? (size_t)1
                                              : logical_cols / thread_count);
      rp.threads_per_row = std::min(logical_cols, thread_count);
      rp.rows_per_thread =
          rp.local_cols > 0 ? regs_per_thread / rp.local_cols : 0;
    } else {
      rp.rows_per_thread = 1;
      rp.local_cols = regs_per_thread;
      rp.threads_per_row = 1;
    }
    return rp;
  }

  // Register index for the reduction inner loop.
  // rv = column iteration variable (0..rp.local_cols-1).
  // row = row variable within thread (0..rp.rows_per_thread-1).
  // The formula is layout-kind-specific and emitted as a C++ string.
  std::string ReduceLocalIndex(const std::string& row, const std::string& rv,
                               size_t local_cols) const {
    if (kind == LayoutKind::WGMMA_ACC) {
      return "(((" + rv + " / 2) * 4) + (" + row + " * 2) + (" + rv + " % 2))";
    }
    if (kind == LayoutKind::UNIFORM) {
      return "(" + row + " * " + std::to_string(local_cols) + " + " + rv + ")";
    }
    return "0";
  }

  // Inverse of ForwardIndex: register index + tid -> logical row index.
  // Used to reconstruct iteration variables inside register-direct automaps
  // when IVs appear in non-fragment expressions.
  std::string LogicalRowFromReg(const std::string& r,
                                const std::string& tid) const {
    switch (kind) {
    case LayoutKind::UNIFORM: {
      std::string flat = LogicalFlatFromReg(r, tid);
      if (logical_cols <= 1) return flat;
      return "(" + flat + " / " + std::to_string(logical_cols) + ")";
    }
    case LayoutKind::WGMMA_ACC:
    case LayoutKind::WGMMA_RS_A:
      return "(" + tid + " / 32 * 16 + ((" + r + ") % 4) / 2 * 8 + " + tid +
             " % 32 / 4)";
    case LayoutKind::REPLICATED_1D:
      if (reduce_row_stride > 0) {
        return "(" + r + " * " + std::to_string(reduce_row_stride) + " + " +
               tid + " / " + std::to_string(replicate) + ")";
      }
      return "(" + tid + " / 32 * 16 + ((" + r + ") % 4) / 2 * 8 + " + tid +
             " % 32 / 4)";
    default: return "0";
    }
  }

  // Inverse of ForwardIndex: register index + tid -> logical column index.
  // Only meaningful for 2D layouts.
  std::string LogicalColFromReg(const std::string& r,
                                const std::string& tid) const {
    switch (kind) {
    case LayoutKind::UNIFORM: {
      if (logical_cols <= 1) return "0";
      std::string flat = LogicalFlatFromReg(r, tid);
      return "(" + flat + " % " + std::to_string(logical_cols) + ")";
    }
    case LayoutKind::WGMMA_ACC:
    case LayoutKind::WGMMA_RS_A:
      return "((" + r + ") / 4 * 8 + " + tid + " % 32 % 4 * 2 + (" + r +
             ") % 2)";
    default: return "0";
    }
  }

  // Flat logical index from register index (for 1D or flat 2D access).
  // UNIFORM 1D: flat = __r * thread_count + tid
  // UNIFORM 2D: same formula (row-major flat)
  std::string LogicalFlatFromReg(const std::string& r,
                                 const std::string& tid) const {
    switch (kind) {
    case LayoutKind::UNIFORM:
      if (vec_width > 1) {
        std::string W = std::to_string(vec_width);
        std::string TW = std::to_string(thread_count * vec_width);
        return "((" + r + ") / " + W + " * " + TW + " + " + tid + " * " + W +
               " + (" + r + ") % " + W + ")";
      }
      return "(" + r + " * " + thread_count_expr + " + " + tid + ")";
    case LayoutKind::WGMMA_ACC:
    case LayoutKind::WGMMA_RS_A: {
      auto row = LogicalRowFromReg(r, tid);
      auto col = LogicalColFromReg(r, tid);
      return "(" + row + " * " + std::to_string(logical_cols) + " + " + col +
             ")";
    }
    case LayoutKind::REPLICATED_1D: return LogicalRowFromReg(r, tid);
    default: return "0";
    }
  }

  bool IsCompatible(const FragmentLayout& other) const {
    auto wgmma_pair = [&](LayoutKind a, LayoutKind b) {
      return (a == LayoutKind::WGMMA_ACC && b == LayoutKind::WGMMA_RS_A) ||
             (a == LayoutKind::WGMMA_RS_A && b == LayoutKind::WGMMA_ACC);
    };
    if (kind != other.kind && !wgmma_pair(kind, other.kind)) return false;
    if (kind == LayoutKind::UNIFORM)
      return thread_count == other.thread_count && vec_width == other.vec_width;
    return logical_rows == other.logical_rows &&
           logical_cols == other.logical_cols;
  }

  bool IsMMAAnchored() const {
    return kind == LayoutKind::WGMMA_ACC || kind == LayoutKind::CTMMA_ACC ||
           kind == LayoutKind::WGMMA_RS_A;
  }

  // -------------------------------------------------------------------------
  // Factory methods
  // -------------------------------------------------------------------------

  static FragmentLayout MakeUniform(size_t total_elements, size_t threads,
                                    const std::string& thread_expr) {
    FragmentLayout fl;
    fl.kind = LayoutKind::UNIFORM;
    fl.thread_count = threads;
    fl.thread_count_expr = thread_expr;
    fl.regs_per_thread =
        threads > 0 ? (total_elements + threads - 1) / threads : total_elements;
    fl.logical_rows = total_elements;
    fl.logical_cols = 1;
    if (fl.regs_per_thread >= 4 && fl.regs_per_thread % 4 == 0 && threads > 0 &&
        total_elements % (threads * 4) == 0)
      fl.vec_width = 4;
    return fl;
  }

  static FragmentLayout MakeUniform2D(size_t rows, size_t cols, size_t threads,
                                      const std::string& thread_expr) {
    FragmentLayout fl;
    fl.kind = LayoutKind::UNIFORM;
    fl.thread_count = threads;
    fl.thread_count_expr = thread_expr;
    size_t total = rows * cols;
    fl.regs_per_thread = threads > 0 ? (total + threads - 1) / threads : total;
    fl.logical_rows = rows;
    fl.logical_cols = cols;
    if (fl.regs_per_thread >= 4 && fl.regs_per_thread % 4 == 0 && threads > 0 &&
        total % (threads * 4) == 0)
      fl.vec_width = 4;
    return fl;
  }

  // WGMMA accumulator: M=64 rows, N columns, 128 threads.
  // regs_per_thread = N/2 for f32 accumulator (groups of 4 regs, N/8 groups).
  static FragmentLayout MakeWGMMA_ACC(size_t M, size_t N) {
    FragmentLayout fl;
    fl.kind = LayoutKind::WGMMA_ACC;
    fl.thread_count = 128;
    fl.thread_count_expr = "128";
    fl.logical_rows = M;
    fl.logical_cols = N;
    fl.warp_count = 4;
    fl.rows_per_thread = 2;
    fl.threads_per_row = 4;
    fl.cols_per_group = N / 8;
    fl.regs_per_thread = N / 2;
    return fl;
  }

  // WGMMA RS-mode A operand. For now treat identically to WGMMA_ACC since
  // the fp16 RS A register layout matches the accumulator column layout.
  static FragmentLayout MakeWGMMA_RS_A(size_t M, size_t K) {
    FragmentLayout fl;
    fl.kind = LayoutKind::WGMMA_RS_A;
    fl.thread_count = 128;
    fl.thread_count_expr = "128";
    fl.logical_rows = M;
    fl.logical_cols = K;
    fl.warp_count = 4;
    fl.rows_per_thread = 2;
    fl.threads_per_row = 4;
    fl.cols_per_group = K / 8;
    fl.regs_per_thread = K / 2;
    return fl;
  }

  // 1D reduction target derived from a 2D accumulator layout.
  // Collapses column dimension; replicate = threads sharing the same row.
  static FragmentLayout MakeReplicated1D(size_t N,
                                         const FragmentLayout& src_acc) {
    FragmentLayout fl;
    fl.kind = LayoutKind::REPLICATED_1D;
    fl.thread_count = src_acc.thread_count;
    fl.thread_count_expr = src_acc.thread_count_expr;
    fl.logical_rows = N;
    fl.logical_cols = 1;
    fl.rows_per_thread = src_acc.rows_per_thread;
    fl.replicate = src_acc.threads_per_row;
    fl.regs_per_thread = fl.rows_per_thread;
    return fl;
  }

  // Reduction destination derived from any 2D source layout.
  // Uses GetReduceParams() to derive REPLICATED_1D layout for the target.
  static FragmentLayout MakeReduceDst(const FragmentLayout& src) {
    if (src.kind == LayoutKind::WGMMA_ACC) {
      return MakeReplicated1D(src.logical_rows, src);
    }
    auto rp = src.GetReduceParams();
    FragmentLayout fl;
    fl.kind = LayoutKind::REPLICATED_1D;
    fl.thread_count = src.thread_count;
    fl.thread_count_expr = src.thread_count_expr;
    fl.logical_rows = src.logical_rows;
    fl.logical_cols = 1;
    fl.rows_per_thread = rp.rows_per_thread;
    fl.threads_per_row = rp.threads_per_row;
    fl.replicate = rp.threads_per_row;
    fl.regs_per_thread = rp.rows_per_thread;
    // For non-WGMMA: row = r * stride + tid / replicate
    fl.reduce_row_stride =
        fl.replicate > 0 ? fl.thread_count / fl.replicate : 0;
    return fl;
  }

  // SM80 mma.sync accumulator: M*N elements across 32 threads.
  static FragmentLayout MakeCTMMA_ACC(size_t M, size_t N) {
    FragmentLayout fl;
    fl.kind = LayoutKind::CTMMA_ACC;
    fl.thread_count = 32;
    fl.thread_count_expr = "32";
    fl.logical_rows = M;
    fl.logical_cols = N;
    fl.warp_count = 1;
    fl.regs_per_thread = M * N / 32;
    return fl;
  }
};

} // namespace Choreo

#endif // __CHOREO_FRAGMENT_LAYOUT_HPP__
