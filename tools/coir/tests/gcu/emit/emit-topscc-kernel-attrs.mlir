// RUN: coir-opt --coir-emit-topscc %s | FileCheck %s
//
// Test that kernel attributes (launch_bounds, maxnreg) emit as comments
// on topscc since this target does not support them.

module attributes { "coir.target" = "topscc", "coir.arch" = "gcu300" } {

// CHECK: __device__
// CHECK-SAME: /* launch_bounds(256, 2) unsupported */
// CHECK-SAME: /* maxnreg(32) unsupported */
// CHECK-SAME: void
coir.kernel @kern_with_attrs(
    %a: !coir.tensor<128xf32, global>)
    attributes {
      launchBounds = #coir.launch_bounds<256, 2>,
      maxNreg = #coir.maxnreg<32>
    } {
  coir.parallel (%pid) in [1] level = #coir.level<block> {
    coir.yield
  }
  coir.return
}

}
