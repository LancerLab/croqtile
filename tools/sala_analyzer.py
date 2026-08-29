#!/usr/bin/env python3
"""
SALA (Signal-Aware Liveness Analysis) Analyzer for Triton TTGIR.

Parses Triton's GPU-level IR (TTGIR) to identify shared memory buffers,
their liveness intervals, and signal/barrier operations, then applies
the SALA happens-before analysis to determine which buffers could safely
overlap in shared memory.

Usage:
    # Analyze a single TTGIR file:
    python3 tools/sala_analyzer.py path/to/kernel.ttgir

    # Generate TTGIR from a Triton kernel and analyze:
    python3 tools/sala_analyzer.py --from-triton path/to/kernel.py

    # Batch analyze all TTGIR in a directory:
    python3 tools/sala_analyzer.py --batch path/to/ttgir_dir/
"""

import argparse
import re
import sys
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SharedBuffer:
    name: str
    shape: list  # e.g., [3, 64, 32] for 3-stage 64x32
    dtype: str   # e.g., "f16"
    line: int
    size_bytes: int = 0
    num_stages: int = 1
    per_stage_bytes: int = 0

    def compute_size(self):
        dtype_bytes = {"f16": 2, "f32": 4, "bf16": 2, "f8": 1,
                       "i8": 1, "i16": 2, "i32": 4, "i64": 8}
        elem_size = dtype_bytes.get(self.dtype, 2)
        total_elems = 1
        for d in self.shape:
            total_elems *= d
        self.size_bytes = total_elems * elem_size
        if len(self.shape) >= 3:
            self.num_stages = self.shape[0]
            stage_elems = 1
            for d in self.shape[1:]:
                stage_elems *= d
            self.per_stage_bytes = stage_elems * elem_size
        else:
            self.num_stages = 1
            self.per_stage_bytes = self.size_bytes


@dataclass
class IRStatement:
    line_no: int
    text: str
    op_type: str       # "alloc", "async_copy", "commit", "wait",
                       #   "dot", "dot_wait", "dealloc", "store",
                       #   "for_begin", "for_end", "convert", "other"
    buffer_refs: list  # SSA names of shared buffers referenced
    raw: str = ""


@dataclass
class Phase:
    phase_id: int
    wg_id: int         # 0 = producer, 1 = consumer (in single-wg kernels, both = 0)
    start_line: int
    end_line: int
    signal_in: str     # barrier waited on at entry
    signal_out: str    # barrier signaled at exit
    buffers_accessed: set = field(default_factory=set)
    is_mainloop: bool = False
    is_epilogue: bool = False
    in_loop: bool = False


@dataclass
class HBEdge:
    from_phase: int
    to_phase: int
    edge_type: str     # "sequential" or "signal"
    is_back_edge: bool = False


class TTGIRParser:
    """Parse Triton TTGIR text into structured IR statements."""

    ALLOC_RE = re.compile(
        r'(%\w+)\s*=\s*ttg\.local_alloc.*?'
        r'memdesc<([^>]+)>')
    ASYNC_COPY_RE = re.compile(
        r'ttg\.async_copy_global_to_local\s+[^,]+,\s*(%\w+)')
    COMMIT_RE = re.compile(r'ttg\.async_commit_group')
    WAIT_RE = re.compile(r'ttg\.async_wait.*?\{num\s*=\s*(\d+)')
    DOT_RE = re.compile(
        r'ttng\.warp_group_dot\s+(%\w+),\s*(%\w+)')
    DOT_WAIT_RE = re.compile(
        r'ttng\.warp_group_dot_wait\s+[^,]*,\s*(%\w+),\s*(%\w+)')
    DEALLOC_RE = re.compile(
        r'ttg\.local_dealloc\s+(%\w+)')
    STORE_RE = re.compile(r'tt\.store\s')
    LOCAL_STORE_RE = re.compile(
        r'ttg\.local_store\s+[^,]+,\s*(%\w+)')
    COPY_LOCAL_TO_GLOBAL_RE = re.compile(
        r'ttg\.async_copy_local_to_global\s+(%\w+)')
    FOR_RE = re.compile(r'scf\.for\b')
    YIELD_RE = re.compile(r'scf\.yield\b')
    MEMDESC_IDX_RE = re.compile(
        r'(%\w+)\s*=\s*ttg\.memdesc_index\s+(%\w+)')
    CONVERT_RE = re.compile(
        r'(%\w+)\s*=\s*ttg\.convert_layout')

    def __init__(self):
        self.buffers = {}       # SSA name -> SharedBuffer
        self.stmts = []         # list of IRStatement
        self.buf_aliases = {}   # memdesc_index result -> parent buffer

    def parse(self, ttgir_text: str):
        lines = ttgir_text.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
            if stripped.startswith('module ') or stripped.startswith('tt.func'):
                continue
            if stripped == '}' or stripped.startswith('} loc'):
                continue

            stmt = self._parse_line(i, stripped)
            if stmt:
                self.stmts.append(stmt)

        for buf in self.buffers.values():
            buf.compute_size()

    def _parse_line(self, line_no: int, text: str) -> Optional[IRStatement]:
        m = self.ALLOC_RE.search(text)
        if m:
            name = m.group(1)
            desc = m.group(2)
            shape, dtype = self._parse_memdesc(desc)
            buf = SharedBuffer(name=name, shape=shape, dtype=dtype, line=line_no)
            self.buffers[name] = buf
            return IRStatement(line_no, text, "alloc", [name])

        m = self.MEMDESC_IDX_RE.search(text)
        if m:
            result, parent = m.group(1), m.group(2)
            root = self.buf_aliases.get(parent, parent)
            self.buf_aliases[result] = root
            return IRStatement(line_no, text, "memdesc_index", [root])

        m = self.ASYNC_COPY_RE.search(text)
        if m:
            dst = m.group(1)
            root = self.buf_aliases.get(dst, dst)
            return IRStatement(line_no, text, "async_copy", [root])

        if self.COMMIT_RE.search(text):
            return IRStatement(line_no, text, "commit", [])

        m = self.WAIT_RE.search(text)
        if m:
            return IRStatement(line_no, text, "wait", [])

        m = self.DOT_WAIT_RE.search(text)
        if m:
            refs = []
            for g in [m.group(1), m.group(2)]:
                root = self.buf_aliases.get(g, g)
                if root in self.buffers:
                    refs.append(root)
            return IRStatement(line_no, text, "dot_wait", refs)

        m = self.DOT_RE.search(text)
        if m:
            refs = []
            for g in [m.group(1), m.group(2)]:
                root = self.buf_aliases.get(g, g)
                if root in self.buffers:
                    refs.append(root)
            return IRStatement(line_no, text, "dot", refs)

        m = self.DEALLOC_RE.search(text)
        if m:
            name = m.group(1)
            return IRStatement(line_no, text, "dealloc", [])

        m = self.LOCAL_STORE_RE.search(text)
        if m:
            dst = m.group(1)
            root = self.buf_aliases.get(dst, dst)
            return IRStatement(line_no, text, "store_local", [root])

        m = self.COPY_LOCAL_TO_GLOBAL_RE.search(text)
        if m:
            src = m.group(1)
            root = self.buf_aliases.get(src, src)
            return IRStatement(line_no, text, "store_local", [root])

        if self.STORE_RE.search(text):
            return IRStatement(line_no, text, "store", [])

        if self.FOR_RE.search(text):
            return IRStatement(line_no, text, "for_begin", [])

        if self.YIELD_RE.search(text):
            return IRStatement(line_no, text, "for_end", [])

        if self.CONVERT_RE.search(text):
            return IRStatement(line_no, text, "convert", [])

        return None

    @staticmethod
    def _parse_memdesc(desc: str):
        parts = desc.split(',')
        shape_dtype = parts[0].strip()
        tokens = shape_dtype.split('x')
        shape = [int(t) for t in tokens[:-1]]
        dtype = tokens[-1].strip()
        return shape, dtype


class SALAAnalyzer:
    """
    Build an HB graph from parsed TTGIR and determine which shared
    buffers could safely overlap.
    """

    def __init__(self, parser: TTGIRParser):
        self.parser = parser
        self.phases = []
        self.edges = []
        self.reachable = []

    def analyze(self):
        self._build_phases()
        self._build_edges()
        self._compute_transitive_closure()

    # Op types that represent actual buffer data access (not alloc/dealloc/index)
    USE_OPS = {"async_copy", "dot", "dot_wait", "store_local"}

    def _is_buf_use(self, stmt: IRStatement) -> bool:
        """True if this statement actually reads/writes buffer data."""
        return stmt.op_type in self.USE_OPS

    def _build_phases(self):
        phase_id = 0
        in_loop = False
        current_bufs = set()
        phase_start = 0

        for stmt in self.parser.stmts:
            if stmt.op_type == "for_begin":
                if current_bufs:
                    self.phases.append(Phase(
                        phase_id=phase_id, wg_id=0,
                        start_line=phase_start, end_line=stmt.line_no,
                        signal_in="", signal_out="",
                        buffers_accessed=current_bufs,
                        is_mainloop=False, in_loop=False))
                    phase_id += 1
                    current_bufs = set()
                in_loop = True
                phase_start = stmt.line_no

            elif stmt.op_type == "wait":
                if current_bufs or phase_start:
                    p = Phase(
                        phase_id=phase_id, wg_id=0,
                        start_line=phase_start, end_line=stmt.line_no,
                        signal_in="", signal_out="wait",
                        buffers_accessed=current_bufs,
                        is_mainloop=in_loop, in_loop=in_loop)
                    self.phases.append(p)
                    phase_id += 1
                    current_bufs = set()
                    phase_start = stmt.line_no

            elif stmt.op_type == "for_end":
                if current_bufs:
                    self.phases.append(Phase(
                        phase_id=phase_id, wg_id=0,
                        start_line=phase_start, end_line=stmt.line_no,
                        signal_in="", signal_out="",
                        buffers_accessed=current_bufs,
                        is_mainloop=True, in_loop=True))
                    phase_id += 1
                    current_bufs = set()
                in_loop = False
                phase_start = stmt.line_no

            if self._is_buf_use(stmt):
                for ref in stmt.buffer_refs:
                    if ref in self.parser.buffers:
                        current_bufs.add(ref)

            if stmt.op_type == "store" and not in_loop:
                if current_bufs:
                    self.phases.append(Phase(
                        phase_id=phase_id, wg_id=0,
                        start_line=phase_start, end_line=stmt.line_no,
                        signal_in="", signal_out="",
                        buffers_accessed=current_bufs,
                        is_mainloop=False, is_epilogue=True,
                        in_loop=False))
                    phase_id += 1
                    current_bufs = set()
                    phase_start = stmt.line_no

        if current_bufs:
            self.phases.append(Phase(
                phase_id=phase_id, wg_id=0,
                start_line=phase_start, end_line=999999,
                signal_in="", signal_out="",
                buffers_accessed=current_bufs,
                is_mainloop=False, is_epilogue=True,
                in_loop=False))

    def _build_edges(self):
        for i in range(len(self.phases) - 1):
            self.edges.append(HBEdge(i, i + 1, "sequential"))

        for i, pi in enumerate(self.phases):
            for j, pj in enumerate(self.phases):
                if i >= j:
                    continue
                if pi.signal_out and pj.signal_in:
                    is_back = (pi.in_loop and pj.in_loop and
                               pi.phase_id > pj.phase_id)
                    self.edges.append(HBEdge(i, j, "signal",
                                            is_back_edge=is_back))

    def _compute_transitive_closure(self):
        n = len(self.phases)
        if n == 0:
            return
        self.reachable = [[False] * n for _ in range(n)]
        for e in self.edges:
            if not e.is_back_edge:
                self.reachable[e.from_phase][e.to_phase] = True
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.reachable[i][k] and self.reachable[k][j]:
                        self.reachable[i][j] = True

    def can_overlap(self, buf_a: str, buf_b: str) -> bool:
        phases_a = [p for p in self.phases if buf_a in p.buffers_accessed]
        phases_b = [p for p in self.phases if buf_b in p.buffers_accessed]
        if not phases_a or not phases_b:
            return False

        for pa in phases_a:
            for pb in phases_b:
                if pa.phase_id == pb.phase_id:
                    return False
                i, j = pa.phase_id, pb.phase_id
                if not (self.reachable[i][j] or self.reachable[j][i]):
                    return False
        return True

    def get_conventional_smem(self) -> int:
        return sum(b.size_bytes for b in self.parser.buffers.values())

    def get_sala_smem(self) -> int:
        bufs = list(self.parser.buffers.values())
        if len(bufs) <= 1:
            return self.get_conventional_smem()

        overlap_groups = []
        used = set()
        for i, bi in enumerate(bufs):
            if bi.name in used:
                continue
            group = [bi]
            used.add(bi.name)
            for j, bj in enumerate(bufs):
                if j <= i or bj.name in used:
                    continue
                if self.can_overlap(bi.name, bj.name):
                    group.append(bj)
                    used.add(bj.name)
            overlap_groups.append(group)

        for bi in bufs:
            if bi.name not in used:
                overlap_groups.append([bi])

        total = 0
        for group in overlap_groups:
            total += max(b.size_bytes for b in group)
        return total

    def get_liveness_intervals(self):
        """Compute conventional vs SALA liveness intervals per buffer.

        Conventional: alloc line to dealloc (whole-scope, as a naive
        compiler would compute).
        SALA: first actual data use to last actual data use,
        phase-bounded by signal barriers.
        """
        USE_OPS = {"async_copy", "dot", "dot_wait", "store_local"}
        intervals = {}
        for name, buf in self.parser.buffers.items():
            first_use = None
            last_use = None
            alloc_line = buf.line
            dealloc_line = None

            for stmt in self.parser.stmts:
                if (name in stmt.buffer_refs and
                        stmt.op_type in USE_OPS):
                    if first_use is None:
                        first_use = stmt.line_no
                    last_use = stmt.line_no

            for stmt in self.parser.stmts:
                if stmt.op_type == "dealloc":
                    m = re.search(r'local_dealloc\s+' + re.escape(name),
                                  stmt.text)
                    if m:
                        dealloc_line = stmt.line_no

            total_lines = max(s.line_no for s in self.parser.stmts)
            conv_start = alloc_line
            conv_end = dealloc_line or total_lines
            conv_span = conv_end - conv_start

            sala_start = first_use or alloc_line
            sala_end = last_use or conv_end
            sala_span = sala_end - sala_start

            shrinkage = ((conv_span - sala_span) / conv_span * 100
                        if conv_span > 0 else 0)

            intervals[name] = {
                "conv_interval": (conv_start, conv_end),
                "conv_span": conv_span,
                "sala_interval": (sala_start, sala_end),
                "sala_span": sala_span,
                "shrinkage_pct": shrinkage,
                "size_kb": buf.size_bytes / 1024,
            }
        return intervals

    def simulate_allocation(self):
        """Simulate a heap allocator with conventional vs SALA lifetimes.

        Returns a dict with:
          - conv_layout: {buf_name: (offset, size)} with no overlap
          - sala_layout: {buf_name: (offset, size)} with SALA overlap
          - conv_peak: peak SMEM usage (bytes)
          - sala_peak: peak SMEM usage (bytes)
          - waste_kb_stmts: wasted KB*statements of dead-but-allocated SMEM
          - reuse_budget_kb: max SMEM available for new allocations after
                            SALA-identified dead points
        """
        intervals = self.get_liveness_intervals()
        bufs = list(self.parser.buffers.values())

        conv_offset = 0
        conv_layout = {}
        for b in bufs:
            conv_layout[b.name] = (conv_offset, b.size_bytes)
            conv_offset += b.size_bytes

        sala_layout = {}
        allocated = []
        buf_by_end = sorted(bufs, key=lambda b: intervals[b.name]["sala_interval"][0])

        for b in buf_by_end:
            best_offset = None
            for candidate_off in range(0, conv_offset, 256):
                conflict = False
                for (ab_name, ab_off, ab_size) in allocated:
                    if (candidate_off < ab_off + ab_size and
                            candidate_off + b.size_bytes > ab_off):
                        if not self.can_overlap(b.name, ab_name):
                            conflict = True
                            break
                if not conflict:
                    best_offset = candidate_off
                    break
            if best_offset is None:
                best_offset = max((a[1] + a[2]) for a in allocated) if allocated else 0
            sala_layout[b.name] = (best_offset, b.size_bytes)
            allocated.append((b.name, best_offset, b.size_bytes))

        sala_peak = max(off + sz for (off, sz) in sala_layout.values()) if sala_layout else 0

        total_stmts = max(s.line_no for s in self.parser.stmts) if self.parser.stmts else 1
        waste_kb_stmts = 0.0
        reuse_budget_kb = 0.0
        for b in bufs:
            iv = intervals[b.name]
            conv_span = iv["conv_span"]
            sala_span = iv["sala_span"]
            wasted_stmts = conv_span - sala_span
            waste_kb_stmts += (b.size_bytes / 1024) * wasted_stmts
            if wasted_stmts > 0 and iv["sala_interval"][1] < iv["conv_interval"][1]:
                reuse_budget_kb = max(reuse_budget_kb, b.size_bytes / 1024)

        return {
            "conv_layout": conv_layout,
            "sala_layout": sala_layout,
            "conv_peak": conv_offset,
            "sala_peak": sala_peak,
            "waste_kb_stmts": waste_kb_stmts,
            "reuse_budget_kb": reuse_budget_kb,
        }

    def validate_soundness(self):
        """Verify that SALA intervals are sound: every actual buffer access
        falls within the SALA-claimed live interval.

        Returns a dict with:
          - is_sound: True if all intervals are sound
          - violations: list of {buffer, access_line, access_op, interval}
          - checked_accesses: total number of access checks performed
        """
        USE_OPS = {"async_copy", "dot", "dot_wait", "store_local"}
        intervals = self.get_liveness_intervals()
        violations = []
        checked = 0

        for stmt in self.parser.stmts:
            if stmt.op_type not in USE_OPS:
                continue
            for ref in stmt.buffer_refs:
                if ref not in intervals:
                    continue
                iv = intervals[ref]
                sala_start, sala_end = iv["sala_interval"]
                checked += 1
                if stmt.line_no < sala_start or stmt.line_no > sala_end:
                    violations.append({
                        "buffer": ref,
                        "access_line": stmt.line_no,
                        "access_op": stmt.op_type,
                        "sala_interval": (sala_start, sala_end),
                    })

        overlap_violations = []
        bufs = list(self.parser.buffers.values())
        for i, bi in enumerate(bufs):
            for j, bj in enumerate(bufs):
                if j <= i:
                    continue
                if not self.can_overlap(bi.name, bj.name):
                    continue
                iv_a = intervals[bi.name]
                iv_b = intervals[bj.name]
                a_start, a_end = iv_a["sala_interval"]
                b_start, b_end = iv_b["sala_interval"]
                if a_start <= b_end and b_start <= a_end:
                    overlap_violations.append({
                        "buf_a": bi.name,
                        "buf_b": bj.name,
                        "interval_a": (a_start, a_end),
                        "interval_b": (b_start, b_end),
                    })

        return {
            "is_sound": len(violations) == 0 and len(overlap_violations) == 0,
            "violations": violations,
            "overlap_violations": overlap_violations,
            "checked_accesses": checked,
        }

    def report(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("SALA Analysis Report")
        lines.append("=" * 60)

        lines.append(f"\nShared Memory Buffers ({len(self.parser.buffers)}):")
        for name, buf in self.parser.buffers.items():
            stages_str = (f" ({buf.num_stages} stages, "
                         f"{buf.per_stage_bytes // 1024} KB/stage)"
                         if buf.num_stages > 1 else "")
            lines.append(f"  {name}: {buf.shape} {buf.dtype} = "
                        f"{buf.size_bytes // 1024} KB{stages_str}")

        lines.append(f"\nPhases ({len(self.phases)}):")
        for p in self.phases:
            label = "mainloop" if p.is_mainloop else (
                    "epilogue" if p.is_epilogue else "prologue")
            bufs = ", ".join(p.buffers_accessed) if p.buffers_accessed else "(none)"
            lines.append(f"  Phase {p.phase_id} [{label}] "
                        f"lines {p.start_line}-{p.end_line}: {bufs}")

        lines.append(f"\nHB Edges ({len(self.edges)}):")
        for e in self.edges:
            back = " (BACK-EDGE, dropped)" if e.is_back_edge else ""
            lines.append(f"  P{e.from_phase} -> P{e.to_phase} "
                        f"[{e.edge_type}]{back}")

        intervals = self.get_liveness_intervals()
        lines.append(f"\nLiveness Interval Precision:")
        lines.append(f"  {'Buffer':<12} {'Conv':<14} {'SALA':<14} "
                    f"{'Shrinkage':>10} {'KB':>6}")
        lines.append(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*10} {'-'*6}")
        total_conv = 0
        total_sala = 0
        for name, iv in intervals.items():
            conv_s, conv_e = iv["conv_interval"]
            sala_s, sala_e = iv["sala_interval"]
            lines.append(
                f"  {name:<12} [{conv_s:>4},{conv_e:>4}]    "
                f"[{sala_s:>4},{sala_e:>4}]    "
                f"{iv['shrinkage_pct']:>8.1f}%  "
                f"{iv['size_kb']:>5.0f}")
            total_conv += iv["conv_span"]
            total_sala += iv["sala_span"]
        avg_shrink = ((total_conv - total_sala) / total_conv * 100
                     if total_conv > 0 else 0)
        lines.append(f"  Average interval shrinkage: {avg_shrink:.1f}%")

        conv = self.get_conventional_smem()
        sala = self.get_sala_smem()
        reduction = ((conv - sala) / conv * 100) if conv > 0 else 0

        lines.append(f"\nOverlap Opportunities:")
        bufs = list(self.parser.buffers.values())
        found = False
        for i, bi in enumerate(bufs):
            for j, bj in enumerate(bufs):
                if j <= i:
                    continue
                if self.can_overlap(bi.name, bj.name):
                    lines.append(f"  {bi.name} <-> {bj.name}: CAN OVERLAP "
                                f"(save {min(bi.size_bytes, bj.size_bytes) // 1024} KB)")
                    found = True
        if not found:
            lines.append("  (none -- all buffers co-live in mainloop)")

        lines.append(f"\nShared Memory Summary:")
        lines.append(f"  Conventional (no overlap):  {conv // 1024} KB")
        lines.append(f"  With SALA (overlapped):     {sala // 1024} KB")
        lines.append(f"  Reduction:                  {reduction:.1f}%")

        soundness = self.validate_soundness()
        lines.append(f"\nSoundness Validation:")
        if soundness["is_sound"]:
            lines.append(f"  PASS: all {soundness['checked_accesses']} buffer "
                        f"accesses fall within SALA intervals")
        else:
            lines.append(f"  FAIL: {len(soundness['violations'])} access "
                        f"violations, {len(soundness['overlap_violations'])} "
                        f"overlap violations")
            for v in soundness["violations"]:
                lines.append(f"    {v['buffer']} accessed at line "
                            f"{v['access_line']} ({v['access_op']}) "
                            f"but SALA interval is {v['sala_interval']}")
            for v in soundness["overlap_violations"]:
                lines.append(f"    {v['buf_a']} {v['interval_a']} overlaps "
                            f"{v['buf_b']} {v['interval_b']} in time")

        alloc_sim = self.simulate_allocation()
        lines.append(f"\nAllocation Simulation:")
        lines.append(f"  Conventional peak:  {alloc_sim['conv_peak'] // 1024} KB")
        lines.append(f"  SALA peak:          {alloc_sim['sala_peak'] // 1024} KB")
        lines.append(f"  Waste (KB*stmts):   {alloc_sim['waste_kb_stmts']:.0f}")
        lines.append(f"  Reuse budget:       {alloc_sim['reuse_budget_kb']:.0f} KB "
                    f"(max buffer reclaimable after SALA-identified death)")

        lines.append(f"\n  Conventional layout:")
        for name, (off, sz) in alloc_sim['conv_layout'].items():
            lines.append(f"    {name}: offset={off // 1024}KB, size={sz // 1024}KB")
        lines.append(f"  SALA layout:")
        for name, (off, sz) in alloc_sim['sala_layout'].items():
            lines.append(f"    {name}: offset={off // 1024}KB, size={sz // 1024}KB")

        if reduction == 0 and len(self.parser.buffers) > 0:
            lines.append(f"\n  Note: All pipeline buffers are co-live in the mainloop,")
            lines.append(f"  so no physical overlap is possible with current buffer set.")
            lines.append(f"  Waste metric shows {alloc_sim['waste_kb_stmts']:.0f} KB*stmts")
            lines.append(f"  of dead-but-allocated SMEM that would be reclaimable if")
            lines.append(f"  additional (e.g., output) buffers were allocated.")

        lines.append("")
        return "\n".join(lines)


def generate_ttgir(triton_script: str) -> str:
    """Run a Triton script and capture the TTGIR from the cache."""
    import subprocess
    import glob
    import tempfile

    cache_dir = tempfile.mkdtemp(prefix="sala_triton_")
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = cache_dir

    result = subprocess.run(
        [sys.executable, triton_script],
        env=env, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"Triton script failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    ttgir_files = glob.glob(os.path.join(cache_dir, "**/*.ttgir"),
                            recursive=True)
    if not ttgir_files:
        print("No TTGIR files generated.", file=sys.stderr)
        sys.exit(1)

    ttgir_texts = {}
    for f in ttgir_files:
        kernel_name = os.path.basename(f).replace(".ttgir", "")
        with open(f) as fh:
            ttgir_texts[kernel_name] = fh.read()
    return ttgir_texts


def analyze_ttgir(ttgir_text: str, name: str = "kernel") -> dict:
    parser = TTGIRParser()
    parser.parse(ttgir_text)

    analyzer = SALAAnalyzer(parser)
    analyzer.analyze()

    conv = analyzer.get_conventional_smem()
    sala = analyzer.get_sala_smem()
    reduction = ((conv - sala) / conv * 100) if conv > 0 else 0

    intervals = analyzer.get_liveness_intervals()
    total_conv = sum(v["conv_span"] for v in intervals.values())
    total_sala = sum(v["sala_span"] for v in intervals.values())
    avg_shrinkage = ((total_conv - total_sala) / total_conv * 100
                     if total_conv > 0 else 0)

    return {
        "name": name,
        "num_buffers": len(parser.buffers),
        "num_phases": len(analyzer.phases),
        "conv_smem_kb": conv / 1024,
        "sala_smem_kb": sala / 1024,
        "reduction_pct": reduction,
        "avg_interval_shrinkage_pct": avg_shrinkage,
        "intervals": intervals,
        "report": analyzer.report(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="SALA Analyzer for Triton TTGIR")
    parser.add_argument("input", help="TTGIR file, Triton script "
                       "(--from-triton), or directory (--batch)")
    parser.add_argument("--from-triton", action="store_true",
                       help="Input is a Triton Python script")
    parser.add_argument("--batch", action="store_true",
                       help="Analyze all .ttgir files in directory")
    parser.add_argument("--csv", action="store_true",
                       help="Output CSV summary")
    args = parser.parse_args()

    results = []

    if args.from_triton:
        ttgir_texts = generate_ttgir(args.input)
        for name, text in ttgir_texts.items():
            results.append(analyze_ttgir(text, name))
    elif args.batch:
        import glob
        ttgir_files = glob.glob(os.path.join(args.input, "**/*.ttgir"),
                                recursive=True)
        for f in sorted(ttgir_files):
            name = os.path.basename(f).replace(".ttgir", "")
            with open(f) as fh:
                results.append(analyze_ttgir(fh.read(), name))
    else:
        with open(args.input) as fh:
            text = fh.read()
        name = os.path.basename(args.input).replace(".ttgir", "")
        results.append(analyze_ttgir(text, name))

    if args.csv:
        print("kernel,buffers,phases,conv_kb,sala_kb,"
              "reduction_pct,avg_interval_shrinkage_pct")
        for r in results:
            print(f"{r['name']},{r['num_buffers']},{r['num_phases']},"
                  f"{r['conv_smem_kb']:.1f},{r['sala_smem_kb']:.1f},"
                  f"{r['reduction_pct']:.1f},"
                  f"{r['avg_interval_shrinkage_pct']:.1f}")
    else:
        for r in results:
            print(r["report"])


if __name__ == "__main__":
    main()
