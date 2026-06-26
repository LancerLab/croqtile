// CoIRAttrs.td -- Attribute and enum definitions for the CoIR dialect
#ifndef COIR_ATTRS_TD
#define COIR_ATTRS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"

include "CoIR/CoIRDialect.td"

//===----------------------------------------------------------------------===//
// Enum helper classes
//===----------------------------------------------------------------------===//

class CoIR_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::coir";
}

class CoIR_I32EnumAttr<string mnemonic, CoIR_I32Enum enumInfo> :
    EnumAttr<CoIR_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ParallelLevel -- execution hierarchy in GPU parallel model
//===----------------------------------------------------------------------===//

def CoIR_PL_Thread    : I32EnumAttrCase<"THREAD",    0, "thread">;
def CoIR_PL_Group     : I32EnumAttrCase<"GROUP",     1, "group">;
def CoIR_PL_GroupX4   : I32EnumAttrCase<"GROUPx4",   2, "groupx4">;
def CoIR_PL_Block     : I32EnumAttrCase<"BLOCK",     3, "block">;
def CoIR_PL_Cluster   : I32EnumAttrCase<"CLUSTER",   4, "cluster">;
def CoIR_PL_Device    : I32EnumAttrCase<"DEVICE",    5, "device">;
def CoIR_PL_Seq       : I32EnumAttrCase<"SEQ",       6, "seq">;

def CoIR_ParallelLevelEnum : CoIR_I32Enum<
  "ParallelLevel", "parallel execution level", [
    CoIR_PL_Thread,
    CoIR_PL_Group,
    CoIR_PL_GroupX4,
    CoIR_PL_Block,
    CoIR_PL_Cluster,
    CoIR_PL_Device,
    CoIR_PL_Seq
  ]>;

def CoIR_ParallelLevelAttr :
  CoIR_I32EnumAttr<"level", CoIR_ParallelLevelEnum>;

//===----------------------------------------------------------------------===//
// TensorMemorySpace -- target-agnostic memory hierarchy for tensor types
//===----------------------------------------------------------------------===//

def CoIR_TMS_Global   : I32EnumAttrCase<"Global",   0, "global">;
def CoIR_TMS_Shared   : I32EnumAttrCase<"Shared",   1, "shared">;
def CoIR_TMS_Local    : I32EnumAttrCase<"Local",     2, "local">;
def CoIR_TMS_Register : I32EnumAttrCase<"Register", 3, "register">;

def CoIR_TensorMemorySpaceEnum : CoIR_I32Enum<
  "TensorMemorySpace", "tensor memory hierarchy level", [
    CoIR_TMS_Global,
    CoIR_TMS_Shared,
    CoIR_TMS_Local,
    CoIR_TMS_Register
  ]>;

def CoIR_TensorMemorySpaceAttr :
  CoIR_I32EnumAttr<"tensor_memory_space", CoIR_TensorMemorySpaceEnum>;

//===----------------------------------------------------------------------===//
// MMALayout -- operand layout for matrix multiply-accumulate
//===----------------------------------------------------------------------===//

def CoIR_MMA_RowCol : I32EnumAttrCase<"RowCol", 0, "row_col">;
def CoIR_MMA_RowRow : I32EnumAttrCase<"RowRow", 1, "row_row">;
def CoIR_MMA_ColCol : I32EnumAttrCase<"ColCol", 2, "col_col">;
def CoIR_MMA_ColRow : I32EnumAttrCase<"ColRow", 3, "col_row">;

def CoIR_MMALayoutEnum : CoIR_I32Enum<
  "MMALayout", "MMA operand layout", [
    CoIR_MMA_RowCol,
    CoIR_MMA_RowRow,
    CoIR_MMA_ColCol,
    CoIR_MMA_ColRow
  ]>;

def CoIR_MMALayoutAttr :
  CoIR_I32EnumAttr<"mma_layout", CoIR_MMALayoutEnum>;

//===----------------------------------------------------------------------===//
// DMAKind -- transfer operation kind for DMA descriptor pipeline
//===----------------------------------------------------------------------===//

def CoIR_DK_Copy      : I32EnumAttrCase<"Copy",      0, "copy">;
def CoIR_DK_Slice     : I32EnumAttrCase<"Slice",     1, "slice">;
def CoIR_DK_Transpose : I32EnumAttrCase<"Transpose", 2, "transpose">;
def CoIR_DK_Pad       : I32EnumAttrCase<"Pad",       3, "pad">;

def CoIR_DMAKindEnum : CoIR_I32Enum<
  "DMAKind", "DMA transfer operation kind", [
    CoIR_DK_Copy,
    CoIR_DK_Slice,
    CoIR_DK_Transpose,
    CoIR_DK_Pad
  ]>;

def CoIR_DMAKindAttr :
  CoIR_I32EnumAttr<"dma_kind", CoIR_DMAKindEnum>;

//===----------------------------------------------------------------------===//
// AssertSite -- emission tier for runtime safety assertions
//
// Target-agnostic: applies to all backends (GPU, accelerators, etc.).
// The HoistAssertions pass promotes assertions up the tier ladder;
// emitters map each tier to the appropriate host/device check API.
//===----------------------------------------------------------------------===//

def CoIR_AS_Entry : I32EnumAttrCase<"ENTRY", 0, "entry">;
def CoIR_AS_Hoist : I32EnumAttrCase<"HOIST", 1, "hoist">;
def CoIR_AS_Use   : I32EnumAttrCase<"USE",   2, "use">;

def CoIR_AssertSiteEnum : CoIR_I32Enum<
  "AssertSite", "assertion emission site tier", [
    CoIR_AS_Entry,
    CoIR_AS_Hoist,
    CoIR_AS_Use
  ]>;

def CoIR_AssertSiteAttr :
  CoIR_I32EnumAttr<"assert_site", CoIR_AssertSiteEnum>;

//===----------------------------------------------------------------------===//
// AssertUsage -- semantic category of a runtime safety check
//
// Target-agnostic classification. Each category maps to a family of
// predicates that the frontend (SemaChecker) or target adaptor generates.
// Backends and diagnostic tools use this to filter, report, or suppress
// specific check families.
//===----------------------------------------------------------------------===//

def CoIR_AU_Unclassified    : I32EnumAttrCase<"UNCLASSIFIED",    0, "unclassified">;
def CoIR_AU_ShapeCompat     : I32EnumAttrCase<"SHAPE_COMPAT",    1, "shape_compat">;
def CoIR_AU_ElementAccess   : I32EnumAttrCase<"ELEMENT_ACCESS",  2, "element_access">;
def CoIR_AU_LoopBound       : I32EnumAttrCase<"LOOP_BOUND",      3, "loop_bound">;
def CoIR_AU_HwConstraint    : I32EnumAttrCase<"HW_CONSTRAINT",   4, "hw_constraint">;

def CoIR_AssertUsageEnum : CoIR_I32Enum<
  "AssertUsage", "semantic category of safety check", [
    CoIR_AU_Unclassified,
    CoIR_AU_ShapeCompat,
    CoIR_AU_ElementAccess,
    CoIR_AU_LoopBound,
    CoIR_AU_HwConstraint
  ]>;

def CoIR_AssertUsageAttr :
  CoIR_I32EnumAttr<"assert_usage", CoIR_AssertUsageEnum>;

//===----------------------------------------------------------------------===//
// AssertCost -- cost classification for runtime assertion filtering
//
// Target-agnostic cost tier derived from the assertion's nesting depth.
// EstimateAssertCost computes a raw numeric cost (product of enclosing
// loop trip counts) and maps it to one of these tiers. Emitters use the
// `enabled` attribute (computed from the user's --runtime-check threshold)
// to decide whether to emit the assertion.
//===----------------------------------------------------------------------===//

def CoIR_AC_Entry  : I32EnumAttrCase<"ENTRY",  0, "entry">;
def CoIR_AC_Low    : I32EnumAttrCase<"LOW",    1, "low">;
def CoIR_AC_Medium : I32EnumAttrCase<"MEDIUM", 2, "medium">;
def CoIR_AC_High   : I32EnumAttrCase<"HIGH",   3, "high">;

def CoIR_AssertCostEnum : CoIR_I32Enum<
  "AssertCost", "assertion cost classification tier", [
    CoIR_AC_Entry,
    CoIR_AC_Low,
    CoIR_AC_Medium,
    CoIR_AC_High
  ]>;

def CoIR_AssertCostAttr :
  CoIR_I32EnumAttr<"cost_class", CoIR_AssertCostEnum>;

//===----------------------------------------------------------------------===//
// AtomicKind -- read-modify-write operation kind for atomic ops
//===----------------------------------------------------------------------===//

def CoIR_AK_Add  : I32EnumAttrCase<"Add",  0, "add">;
def CoIR_AK_Sub  : I32EnumAttrCase<"Sub",  1, "sub">;
def CoIR_AK_Exch : I32EnumAttrCase<"Exch", 2, "exch">;
def CoIR_AK_Min  : I32EnumAttrCase<"Min",  3, "min">;
def CoIR_AK_Max  : I32EnumAttrCase<"Max",  4, "max">;
def CoIR_AK_And  : I32EnumAttrCase<"And",  5, "and">;
def CoIR_AK_Or   : I32EnumAttrCase<"Or",   6, "or">;
def CoIR_AK_Xor  : I32EnumAttrCase<"Xor",  7, "xor">;
def CoIR_AK_CAS  : I32EnumAttrCase<"CAS",  8, "cas">;

def CoIR_AtomicKindEnum : CoIR_I32Enum<
  "AtomicKind", "atomic read-modify-write operation kind", [
    CoIR_AK_Add,
    CoIR_AK_Sub,
    CoIR_AK_Exch,
    CoIR_AK_Min,
    CoIR_AK_Max,
    CoIR_AK_And,
    CoIR_AK_Or,
    CoIR_AK_Xor,
    CoIR_AK_CAS
  ]>;

def CoIR_AtomicKindAttr :
  CoIR_I32EnumAttr<"atomic_kind", CoIR_AtomicKindEnum>;

//===----------------------------------------------------------------------===//
// LaunchBounds -- CUDA/HIP __launch_bounds__ parameters
//===----------------------------------------------------------------------===//

def CoIR_LaunchBoundsAttr : AttrDef<CoIR_Dialect, "LaunchBounds"> {
  let mnemonic = "launch_bounds";
  let summary = "CUDA/HIP __launch_bounds__ parameters";
  let description = [{
    Encodes up to three __launch_bounds__ parameters:
      maxThreadsPerBlock (required),
      minBlocksPerMultiprocessor (optional, default 0 = unset),
      maxBlocksPerCluster (optional, default 0 = unset).
  }];
  let parameters = (ins
    "int64_t":$maxThreadsPerBlock,
    DefaultValuedParameter<"int64_t", "0">:$minBlocksPerMultiprocessor,
    DefaultValuedParameter<"int64_t", "0">:$maxBlocksPerCluster
  );
  let assemblyFormat = "`<` $maxThreadsPerBlock (`,` $minBlocksPerMultiprocessor^ (`,` $maxBlocksPerCluster^)?)? `>`";
}

//===----------------------------------------------------------------------===//
// Legacy layout constraint (kept for existing DMA ops)
//===----------------------------------------------------------------------===//

def CoIR_LayoutAttr : DialectAttr<CoIR_Dialect, CPred<"true">, "layout"> {
  let summary = "layout attribute";
  let description = [{
    Attribute that encodes a layout (e.g. row-major, column-major).
  }];
}

#endif // COIR_ATTRS_TD
