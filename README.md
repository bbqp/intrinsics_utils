Intrinsics Utils
----------------

This repo contains several convenience functions for generating masks,
computing dot products, copying array contents, and other utilities. This
emmerged intially within a project for speeding up a C-implementation
of Smoothlife, and now functions as both a learning tool and a set of
convenient functions to use in other projects.

Currently Supported Instruction Sets
------------------------------------

The code in this repo currently uses intrinsics functions covering MMX, SSE,
SSE2, SSE3 (a.k.a. PNI), AVX, AVX2, AVX512F and AVX512DQ. The `cpu_flags.h`
header in `./include` defines macros that can used to enable/disable
compilation of functions depending on what your processor supports.

For now, we assume that the user's processor at least supports the above-
mentioned instruction sets up to and including AVX2. 

Compiling the shared object library
-----------------------------------

This is currently being tested locally using gcc version 15.2.1, and assumes
the end user is using gcc to compile this library. To generate the library,
run `make setup` to create needed object and library folders followed by
`make intrinsic_utils` to generate the share object library.
