// RUN: coir-opt %s -verify-diagnostics

// coir.tma.copy is rejected by the op verifier when coir.has_tma = false.
module attributes {coir.has_tma = false} {
  coir.kernel @reject_tma_copy(
      %src: !coir.tensor<128x64xf16>,
      %dst: !coir.tensor<128x64xf16, shared>) {
    // expected-error@+1 {{requires TMA support but target does not provide it}}
    %tok = coir.tma.copy %src to %dst
      : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
    coir.wait %tok : !coir.async
  }
}

// coir.tma.copy verifies cleanly when coir.has_tma is absent (lenient): the
// verifier runs during ASTIRGen op construction before StampTargetOnModule
// stamps the attribute, and only fails on an explicit `false`.
module {
  coir.kernel @allow_tma_copy_without_attr(
      %src: !coir.tensor<128x64xf16>,
      %dst: !coir.tensor<128x64xf16, shared>) {
    %tok = coir.tma.copy %src to %dst
      : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
    coir.wait %tok : !coir.async
  }
}
