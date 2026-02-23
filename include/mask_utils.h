#ifndef MASK_UTILS_H
#define MASK_UTILS_H

#include "cpu_flags.h"
#include "constants.h"
#include <immintrin.h>
#include <stdint.h>

//----------------------------------------------------------------------------
// Functions for creating masks.
//----------------------------------------------------------------------------

// SSE* functions.
__m128i _mm_setmask_fromto_epi32(int, int);
__m128i _mm_setmask_fromto_epi64(int, int);
__m128i _mm_set_mask_epi32(int);
__m128i _mm_set_mask_epi64(int);
__m128 _mm_set_mask_ps(int);
__m128d _mm_set_mask_pd(int);

// AVX2 functions.
__m256i _mm256_setmask_fromto_epi32(int, int);
__m256i _mm256_setmask_fromto_epi64(int, int);
__m256i _mm256_set_mask_epi32(int);
__m256i _mm256_set_mask_epi64(int);
__m256 _mm256_set_mask_ps(int);
__m256d _mm256_set_mask_pd(int);

// AVX512 functions.
#ifdef SUPPORTS_AVX512
__mmask16 _mm512_setmask_fromto_epi32(int, int);
__mmask8 _mm512_setmask_fromto_epi64(int, int);
__mmask16 _mm512_set_mask_epi32(int);
__mmask8 _mm512_set_mask_epi64(int);
#endif

#endif
