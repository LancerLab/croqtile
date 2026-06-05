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

def CoIR_PL_Block     : I32EnumAttrCase<"Block",     0, "block">;
def CoIR_PL_Warpgroup : I32EnumAttrCase<"Warpgroup", 1, "warpgroup">;
def CoIR_PL_Warp      : I32EnumAttrCase<"Warp",      2, "warp">;
def CoIR_PL_Thread    : I32EnumAttrCase<"Thread",     3, "thread">;

def CoIR_ParallelLevelEnum : CoIR_I32Enum<
  "ParallelLevel", "parallel execution level", [
    CoIR_PL_Block,
    CoIR_PL_Warpgroup,
    CoIR_PL_Warp,
    CoIR_PL_Thread
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
// Legacy layout constraint (kept for existing DMA ops)
//===----------------------------------------------------------------------===//

def CoIR_LayoutAttr : DialectAttr<CoIR_Dialect, CPred<"true">, "layout"> {
  let summary = "layout attribute";
  let description = [{
    Attribute that encodes a layout (e.g. row-major, column-major).
  }];
}

#endif // COIR_ATTRS_TD
