# Introduction to Choreo Value Numbering
In Choreo, the value numbering process serves to facilitate Shape Inference by either directly inferring shapes known at compile time or generating shape computation expressions.

While many value numbering approaches focus on eliminating redundant computations (expressions that consistently produce the same value), Choreo's value numbering aims to establish a progressive method for simplifying the Choreo domain-specific construct 'mdspan'. In this context, the application of constant propagation, algebraic simplification, and other techniques becomes more essential.

In terms of the implementation method, Choreo utilizes a simple **scope-based** value numbering approach, an extension of the local value numbering method, complemented by a lexical-scoped value number table[1]. Unlike other value numbering methods, Choreo does not necessitate the construction of basic blocks, pruned-SSA, or a compiler backend, as it applies value numbering directly onto the Abstract Syntax Tree (AST). However, to facilitate this approach, certain restrictions must be imposed on Choreo's syntax rules, particularly to prevent value mutation.

## Quick Example
The value numbering process assigns a value number to all entities containing a list of values, including for mdspan and ituple. For example,
```
mdspan a : {1, b, c + 3}
```
The value numbering process would produce values including:
```
   VN #0:  const_1
   VN #1:  b
   VN #2:  c
   VN #3:  const_3
   VN #4:  +:#2:#3
   VN #5:  #0,#1,#4
```
In this illustration, the number after "VN #" is the value number of an expression. I either represents a constant integer value (known at compile time), a symbolic value (like variable 'a', 'b'), or composition of other value numbers.
Here, value number #0 and #3 represents a reference to constant value, where #1, and #2 represents reference to the integer-typed variable "b" and "c". #4 represents the value of an addition expression, which gives the sum of #2 and #3. That is the expression of 'c + 3' when dereferencing all the value numbers. And lastly, value number #5 represents the list of values defined by mdspan 'a' symbolically.
With such a definition, we may translate the indexing operations, e.g. "a(0)", results in value number #0 and thus a constant of 1.

# Implementations
## From Expressions to Value Numbers
In current implement, the value numbering process in a syntax-directed way. It traverse the Abstract Syntax Tree nodes and generate value numbers when the node contains an expression.

The method of generating value numbers is simple. For each expression, it takes the expression as the input, and returns corresponding value number.

More specifically, it:
* Generates a signature from an expression.
* If the signature has already existed, return the associated value number by looking up the value number table.
* Or else, it associates the signature with a new value number in the value number table, and return it.

Notably, for simplicity, current signature is an string of expression. This can be improved by hashing later (like most value numbering process does).

In addition, to handle the symbol definition and symbol reference, it follows the below rules:

* When an expression is to initialize a symbol (variable or partial type), the symbol is assign the same value number of the expression. More specifically, it associate the signature of the *scoped* symbol name with the value number in the value number table.
* When an expression references a symbol, it looks up its corresponding signature in the value number table, and return it.

Note the symbol must use its *scoped name* for signature generation, since identical names are allowed to be defined in different scope.

## Simplifications
Every time the value numbering process generates a signature, it tries to simplify the expression. For example,

```
   int a = 0 + 1;
```
Given the simple integer expression, the value numbering process could generates signatures and associated value number as illustrated:

```
   VN #0:  const_3
   VN #1:  const_8
   VN #2:  +:#0:#1
   Alias a -> #2
```
Here, without optimizations, two constant values #0 and #1, and a new value number #2, which is an addition of #0 and #1 are generated. #2 has the signature of "+:#0:#1". However, considering that addition of constants can be evaluated at compile time, simplification could be performed. The below output showcases the optimized value numbering process:

```
   VN #0:  const_3
   VN #1:  const_8
   VN #2:  const_11
   Alias a -> #2
```
The signature of #2 now represents a constant, which implies the algebraic simplification is applied successfully.

## From Value Numbers to Expression
After value numbering and simplification, each expression is associated with its simplified value number. In Choreo, this is important step for the shape inference process of the mdspan type.

The *mdspan* type is a partial type that defines count of fundamental-typed elements. And in programming, together with a fundamental type, e.g, it decides the memory size an *spanned* object takes.

The shape detail of *mdspan* could be runtime decided. However, in many scenario, the shape is possible to be fully evaluated to fixed numbers at compile time. In such a scenario, static memory allocation can also be triggered at compile time to best utilize the system memory. Or else, the it requires to generate code to calculate the shapes.

Consequently, except the constant values, it is required to generate expressions back from value numbers.

# Design Considerations
## Lexical Scopes and Restriction
In Choreo, we applying the value numbering *GLOBALLY* but on top of AST. As it is a syntax-directed method without basic blocks and thus lack data flow analysis, it is impossible to inference a shape with varying dimensions.

## Limitations
More specifically, it has to set two limitations:
1. Any mdspan can not have element with bounded integer type, or produced from bounded ituple.
2. Simple integer type and ituple type can not be re-defined.



# Reference
[1] Briggs P, Cooper K D, Simpson L T. Value numbering[J]. Software: Practice and Experience, 1997, 27(6): 701-724.
