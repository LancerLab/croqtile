### *Factor* Target Specific Details

In some target like *factor*, it requires the device function to follow C-linkage, where the device functions must be prefixed with `extern "C"`. Since template functions can not be applies to functions with C-linkage, Choreo has some wrapper tricks to enable it. For example:

```choreo
__cok__ {
extern "C" void bar(int v) {};  // v is a compile-time constant
}

__co__ void foo() {
  parallel p by 1 { call bar(3); }
}
```

Programmers can turn `bar` into a function template, like following:

```choreo
__cok__ {
template <int v> void bar() {};

// compiler generate a explicit function template specialization:
// template<> void bar<3>() {};
}

__co__ void foo() {
  parallel p by 1 { call bar<3>(); }
}
```

As the comment depicts, Choreo compiler may generate a specialization version of function template `bar<3>`, which enables the function template as well. But note that, the function template no longer requires `extern "C" decoration` as those for normal *factor* device functions.

As the implementation is different with other targets, you may utilize `-kt` to enable such functionality for *choreo-factor* compilation.

