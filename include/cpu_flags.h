#ifndef SUPPORT_H
#define SUPPORT_H

#ifdef __MMX__
#define SUPPORTS_MMX __builtin_cpu_supports("mmx")
#endif

#ifdef __SSE__
#define SUPPORTS_SSE __builtin_cpu_supports("sse")
#endif

#ifdef __SSE2__
#define SUPPORTS_SSE2 __builtin_cpu_supports("sse2")
#endif

#ifdef __SSE3__
#define SUPPORTS_SSE3 __builtin_cpu_supports("sse3")
#endif

#ifdef __AVX__
#define SUPPORTS_AVX __builtin_cpu_supports("avx")
#endif

#ifdef __AVX2__
#define SUPPORTS_AVX2 __builtin_cpu_supports("avx2")
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define SUPPORTS_AVX512 (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq"))
#endif

#endif
