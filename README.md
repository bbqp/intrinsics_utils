Intrinsics Utils
----------------

Various utilites and convenience functions for working with intrinsic functions.

Currently Support Instruction Sets
----------------------------------

The code in this repo is designed to support MMX, SSE(1,2), AVX, AVX2, AVX512F
and AVX512DQ. The cpu_flags.h header in ./include defines macros for 
 

Compiling the shared object library
-----------------------------------

This is currently being tested locally using gcc version 15.2.1, and assumes
the end user is using gcc. To generate the library, run `make setup` followed
by `make intrinsic_utils`.
