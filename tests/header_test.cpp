#include "utils/choreo.h"

void foo(choreo::f32* a) {
  choreo::spanned<choreo::f32, 2> spanned_a(a, {1, 2});
}

template <typename T, int N>
void bar(const choreo::spanned<T, N> & b) {}

void foobar(float* a) {
  bar(choreo::make_spanned<2>(a, {1, 2}));
}

void fooboo() {
  float a[2][3];
  bar(choreo::make_spanned(a));
}
