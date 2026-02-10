#ifndef MASK_UTILS_H
#define MASK_UTILS_H


#include <immintrin.h>
#include <stdint.h>

#define DOUBLE_PER_M128_REG 2
#define INT64_PER_M128_REG DOUBLE_PER_M128_REG
#define DOUBLE_PER_M256_REG 4
#define INT64_PER_M256_REG DOUBLE_PER_M256_REG

#define FLOAT_PER_M128_REG 4
#define INT32_PER_M128_REG FLOAT_PER_M128_REG
#define FLOAT_PER_M256_REG 8
#define INT32_PER_M256_REG FLOAT_PER_M256_REG

#define INT32_ZERO ((int32_t)0)
#define INT32_ALLBITS ((int32_t)0xFFFFFFFF)
#define INT32_LOWBIT  ((int32_t)0x00000001)
#define INT32_HIGHBIT ((int32_t)0x80000000)

#define INT64_ZERO    ((int64_t)0)
#define INT64_LOWBIT  ((int64_t)0x0000000000000001)
#define INT64_HIGHBIT ((int64_t)0x8000000000000000)
#define INT64_ALLBITS ((int64_t)0xFFFFFFFFFFFFFFFF)


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


#endif
