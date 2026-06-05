// CoIRDialect.td

#ifndef COIR_DIALECT_TD
#define COIR_DIALECT_TD

include "mlir/IR/OpBase.td"

def CoIR_Dialect : Dialect {
  let name = "coir";
  let cppNamespace = "coir";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // COIR_DIALECT_TD
