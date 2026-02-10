#include "mask_utils.h"
#include <immintrin.h>

#define M128_LPERM_MASK (0x39 + ((0x4e) << 8) + ((0x93) << 16) + ((0xe4) << 24))
#define M128_LPERM_TO_IMM8(NPERMS) ((M128_LPERM_MASK >> ((NPERMS) - 1)) & (INT32_ALLBITS))

#define LPERM1 0x39
#define LPERM2 0x4e
#define LPERM3 0x93

#define M128_RPERM_MASK (0x93 + ((0x4e) << 8) + ((0x39) << 16) + ((0xe4) << 24))
#define M128_RPERM_TO_IMM8(NPERMS) ((M128_RPERM_MASK >> ((NPERMS) - 1)) & (INT32_ALLBITS))


//----------------------------------------------------------------------------
// MMX/SSE*-compatible functions for creating integer, single, and double
// masks.
//----------------------------------------------------------------------------

__m128i _mm_setmask_fromto_epi32(int from, int to)
{
	__m128i mask;

	if (from > to || from >= INT32_PER_M128_REG) {
		mask = _mm_set1_epi32(0);
	} else {
		switch (from) {
			case 0:
				switch(to) {
					case 0:
						mask = _mm_setr_epi32(INT32_ALLBITS, 0, 0, 0);
						break;
					case 1:
						mask = _mm_setr_epi32(INT32_ALLBITS, INT32_ALLBITS, 0, 0);
						break;
					case 2:
						mask = _mm_setr_epi32(INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm_set1_epi32(INT32_ALLBITS);
				}
				break;
			case 1:
				switch(to) {
					case 1:
						mask = _mm_setr_epi32(0, INT32_ALLBITS, 0, 0);
						break;
					case 2:
						mask = _mm_setr_epi32(0, INT32_ALLBITS, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm_setr_epi32(0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS);
				}
				break;
			case 2:
				switch(to) {
					case 2:
						mask = _mm_setr_epi32(0, 0, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm_setr_epi32(0, 0, INT32_ALLBITS, INT32_ALLBITS);
				}
				break;
			case 3:
				mask = _mm_setr_epi32(0, 0, 0, INT32_ALLBITS);
		}
	}

	return mask;
}

__m128i _mm_setmask_fromto_epi64(int from, int to)
{
	__m128i mask;

	if (from > to || from >= INT64_PER_M128_REG) {
		mask = _mm_set1_epi64x(0);
	} else {
		switch (from) {
			case 0:
				switch(to) {
					case 0:
						mask = _mm_setr_epi64(_mm_cvtsi64_m64(INT64_ALLBITS), _mm_cvtsi64_m64(0));
						break;
					default:
						mask = _mm_set1_epi64x(INT64_ALLBITS);
				}

				break;
			case 1:
				mask = _mm_setr_epi64(_mm_cvtsi64_m64(0), _mm_cvtsi64_m64(INT64_ALLBITS));
				
				break;
			default:
				mask = _mm_set1_epi64x(0);
		}
	}

	return mask;
}

__m128i _mm_set_mask_epi32(int cutoff)
{
	return _mm_setmask_fromto_epi32(0, cutoff);
}

__m128i _mm_set_mask_epi64(int cutoff)
{
	return _mm_setmask_fromto_epi64(0, cutoff);
}

__m128 _mm_set_mask_ps(int cutoff)
{
	return _mm_castsi128_ps(_mm_set_mask_epi32(cutoff));
}

__m128d _mm_set_mask_pd(int cutoff)
{
	return _mm_castsi128_pd(_mm_set_mask_epi64(cutoff));
}

//----------------------------------------------------------------------------
// AVX*-compatible functions for creating integer, single, and double masks.
//----------------------------------------------------------------------------

__m256i _mm256_setmask_fromto_epi32(int from, int to)
{
	__m256i mask;

	if (from > to || from >= INT32_PER_M256_REG) {
		mask = _mm256_set1_epi32(0);
	} else {
		switch (from) {
			case 0:
				switch(to) {
					case 0:
						mask = _mm256_setr_epi32(INT32_ALLBITS, 0, 0, 0, 0, 0, 0, 0);
						break;
					case 1:
						mask = _mm256_setr_epi32(INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0, 0, 0, 0);
						break;
					case 2:
						mask = _mm256_setr_epi32(INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0, 0, 0);
						break;
					case 3:
						mask = _mm256_setr_epi32(INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0, 0);
						break;
					case 4:
						mask = _mm256_setr_epi32(INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0);
						break;
					case 5:
						mask = _mm256_setr_epi32(INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0);
						break;
					case 6:
						mask = _mm256_setr_epi32(INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm256_set1_epi32(INT32_ALLBITS);
				}
				break;
			case 1:
				switch(to) {
					case 1:
						mask = _mm256_setr_epi32(0, INT32_ALLBITS, 0, 0, 0, 0, 0, 0);
						break;
					case 2:
						mask = _mm256_setr_epi32(0, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0, 0, 0);
						break;
					case 3:
						mask = _mm256_setr_epi32(0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0, 0);
						break;
					case 4:
						mask = _mm256_setr_epi32(0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0);
						break;
					case 5:
						mask = _mm256_setr_epi32(0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0);
						break;
					case 6:
						mask = _mm256_setr_epi32(0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm256_setr_epi32(0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS);
				}
				break;
			case 2:
				switch(to) {
					case 2:
						mask = _mm256_setr_epi32(0, 0, INT32_ALLBITS, 0, 0, 0, 0, 0);
						break;
					case 3:
						mask = _mm256_setr_epi32(0, 0, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0, 0);
						break;
					case 4:
						mask = _mm256_setr_epi32(0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0);
						break;
					case 5:
						mask = _mm256_setr_epi32(0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0);
						break;
					case 6:
						mask = _mm256_setr_epi32(0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm256_setr_epi32(0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS);
				}
				break;
			case 3:
				switch(to) {
					case 3:
						mask = _mm256_setr_epi32(0, 0, 0, INT32_ALLBITS, 0, 0, 0, 0);
						break;
					case 4:
						mask = _mm256_setr_epi32(0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, 0, 0, 0);
						break;
					case 5:
						mask = _mm256_setr_epi32(0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0, 0);
						break;
					case 6:
						mask = _mm256_setr_epi32(0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm256_setr_epi32(0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS);
				}
				break;
			case 4:
				switch(to) {
					case 4:
						mask = _mm256_setr_epi32(0, 0, 0, 0, INT32_ALLBITS, 0, 0, 0);
						break;
					case 5:
						mask = _mm256_setr_epi32(0, 0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, 0, 0);
						break;
					case 6:
						mask = _mm256_setr_epi32(0, 0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm256_setr_epi32(0, 0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS);
				}
				break;
			case 5:
				switch(to) {
					case 5:
						mask = _mm256_setr_epi32(0, 0, 0, 0, 0, INT32_ALLBITS, 0, 0);
						break;
					case 6:
						mask = _mm256_setr_epi32(0, 0, 0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm256_setr_epi32(0, 0, 0, 0, 0, INT32_ALLBITS, INT32_ALLBITS, INT32_ALLBITS);
				}
				break;
			case 6:
				switch(to) {
					case 6:
						mask = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, INT32_ALLBITS, 0);
						break;
					default:
						mask = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, INT32_ALLBITS, INT32_ALLBITS);
				}
				break;
			default:
				mask = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, INT32_ALLBITS);
		}
	}

	return mask;
}

__m256i _mm256_setmask_fromto_epi64(int from, int to)
{
	__m256i mask;

	if (from > to || from >= INT64_PER_M256_REG) {
		mask = _mm256_set1_epi64x(0);
	} else {
		switch (from) {
			case 0:
				switch(to) {
					case 0:
						mask = _mm256_setr_epi64x(INT64_ALLBITS, 0, 0, 0);
						break;
					case 1:
						mask = _mm256_setr_epi64x(INT64_ALLBITS, INT64_ALLBITS, 0, 0);
						break;
					case 2:
						mask = _mm256_setr_epi64x(INT64_ALLBITS, INT64_ALLBITS, INT64_ALLBITS, 0);
						break;
					default:
						mask = _mm256_set1_epi64x(INT64_ALLBITS);
				}
				
				break;
			case 1:
				switch(to) {
					case 1:
						mask = _mm256_setr_epi64x(0, INT64_ALLBITS, 0, 0);
						break;
					case 2:
						mask = _mm256_setr_epi64x(0, INT64_ALLBITS, INT64_ALLBITS, 0);
						break;
					default:
						mask = _mm256_setr_epi64x(0, INT64_ALLBITS, INT64_ALLBITS, INT64_ALLBITS);
				}
				
				break;
			case 2:
				switch(to) {
					case 2:
						mask = _mm256_setr_epi64x(0, 0, INT64_ALLBITS, 0);
						break;
					default:
						mask = _mm256_setr_epi64x(0, 0, INT64_ALLBITS, INT64_ALLBITS);
				}
				
				break;
			case 3:
				mask = _mm256_setr_epi64x(0, 0, 0, INT64_ALLBITS);
				
				break;
			default:
				mask = _mm256_set1_epi64x(0);
		}
	}

	return mask;
}

__m256i _mm256_set_mask_epi32(int cutoff)
{	
	return _mm256_setmask_fromto_epi32(0, cutoff);
}

__m256i _mm256_set_mask_epi64(int cutoff)
{
	return _mm256_setmask_fromto_epi64(0, cutoff);
}

__m256 _mm256_set_mask_ps(int cutoff)
{
	return _mm256_castsi256_ps(_mm256_set_mask_epi32(cutoff));
}

__m256d _mm256_set_mask_pd(int cutoff)
{
	return _mm256_castsi256_pd(_mm256_set_mask_epi64(cutoff));
}

__mmask16 _mm512_setmask_fromto_epi32(int from, int to)
{
    __mmask16 mask; 

	if (from > to || from >= INT32_PER_M256_REG) {
		mask = _mm512_movepi32_mask(_mm512_set1_epi32(0));
	} else {
		mask = _mm512_movepi32_mask(_mm512_set1_epi32(INT32_ALLBITS));
        mask = _kshiftri_mask16(mask, from + 1);
        mask = _kshiftli_mask16(mask, INT32_PER_M512_REG - 1 - to);
    }

	return mask; 
}

__mmask8 _mm512_setmask_fromto_epi64(int from, int to)
{
    __mmask8 mask; 

	if (from > to || from >= INT64_PER_M256_REG) {
		mask = _mm512_movepi64_mask(_mm512_set1_epi64(0));
	} else {
		mask = _mm512_movepi64_mask(_mm512_set1_epi32(INT64_ALLBITS));
        mask = _kshiftri_mask8(mask, from + 1);
        mask = _kshiftli_mask8(mask, INT64_PER_M512_REG - 1 - to);
    }

	return mask; 
}

__mmask16 _mm512_set_mask_epi32(int cutoff)
{
    return _mm512_setmask_fromto_epi32(0, cutoff);
}

__mmask8 _mm512_set_mask_epi64(int cutoff)
{
    return _mm512_setmask_fromto_epi64(0, cutoff);
}
