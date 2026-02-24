
#include "cpu_flags.h"
#include "constants.h"
#include "mask_utils.h"
#include <immintrin.h>

//----------------------------------------------------------------------------
// MMX/SSE*-compatible functions for creating integer, single, and double
// masks.
//----------------------------------------------------------------------------

__m128i _mm_setmask_fromto_epi32(int from, int to)
{
	__m128i mask;

	if (from > to || from > INT32_PER_M128_REG - 1 || to < 0) {
		mask = _mm_set1_epi32(0);
	} else {
        if (from < 0) {
            from = 0;
        }

        if (to > INT32_PER_M128_REG - 1) {
            to = INT32_PER_M128_REG - 1;
        }

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

	if (from > to || from > INT64_PER_M128_REG - 1 || to < 0) {
		mask = _mm_set1_epi64x(0);
	} else {
        if (from < 0) {
            from = 0;
        }

        if (to > INT64_PER_M128_REG - 1) {
            to = INT64_PER_M128_REG - 1;
        }

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

__m128i _mm_set_mask_epi32(int cutoff_index_index)
{
	return _mm_setmask_fromto_epi32(0, cutoff_index_index);
}

__m128i _mm_set_mask_epi64(int cutoff_index)
{
	return _mm_setmask_fromto_epi64(0, cutoff_index);
}

__m128 _mm_set_mask_ps(int cutoff_index)
{
	return _mm_castsi128_ps(_mm_set_mask_epi32(cutoff_index));
}

__m128d _mm_set_mask_pd(int cutoff_index)
{
	return _mm_castsi128_pd(_mm_set_mask_epi64(cutoff_index));
}

//----------------------------------------------------------------------------
// AVX*-compatible functions for creating integer, single, and double masks.
//----------------------------------------------------------------------------

__m256i _mm256_setmask_fromto_epi32(int from, int to)
{
	__m256i mask;

	if (from > to || from > INT32_PER_M256_REG - 1 || to < 0) {
		mask = _mm256_set1_epi32(0);
	} else {
        if (from < 0) {
            from = 0;
        }

        if (to > INT32_PER_M256_REG - 1) {
            to = INT32_PER_M256_REG - 1;
        }

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

	if (from > to || from > INT64_PER_M256_REG - 1 || to < 0) {
		mask = _mm256_set1_epi64x(0);
	} else {
        if (from < 0) {
            from = 0;
        }

        if (to > INT64_PER_M256_REG - 1) {
            to = INT64_PER_M256_REG - 1;
        }

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

__m256i _mm256_set_mask_epi32(int cutoff_index)
{	
	return _mm256_setmask_fromto_epi32(0, cutoff_index);
}

__m256i _mm256_set_mask_epi64(int cutoff_index)
{
	return _mm256_setmask_fromto_epi64(0, cutoff_index);
}

__m256 _mm256_set_mask_ps(int cutoff_index)
{
	return _mm256_castsi256_ps(_mm256_set_mask_epi32(cutoff_index));
}

__m256d _mm256_set_mask_pd(int cutoff_index)
{
	return _mm256_castsi256_pd(_mm256_set_mask_epi64(cutoff_index));
}

#ifdef SUPPORTS_AVX512
__mmask16 _mm512_setmask_fromto_epi32(int from, int to)
{
    __mmask16 mask;
    __mmask16 lmask;
    __mmask16 rmask;

	if (from > to || from > INT32_PER_M512_REG - 1 || to < 0) {
		mask = _mm512_movepi32_mask(_mm512_set1_epi32(0));
	} else {
		mask = _mm512_movepi32_mask(_mm512_set1_epi32(INT32_ALLBITS));

        if (from < 0) {
            from = 0;
        }

        if (to > INT32_PER_M512_REG - 1) {
            to = INT32_PER_M512_REG - 1;
        }

        switch (from) {
            case 0:
                lmask = _kshiftli_mask16(mask, 0);
                break;
            case 1:
                lmask = _kshiftli_mask16(mask, 1);
                break;
            case 2:
                lmask = _kshiftli_mask16(mask, 2);
                break;
            case 3:
                lmask = _kshiftli_mask16(mask, 3);
                break;
            case 4:
                lmask = _kshiftli_mask16(mask, 4);
                break;
            case 5:
                lmask = _kshiftli_mask16(mask, 5);
                break;
            case 6:
                lmask = _kshiftli_mask16(mask, 6);
                break;
            case 7:
                lmask = _kshiftli_mask16(mask, 7);
                break;
            case 8:
                lmask = _kshiftli_mask16(mask, 8);
                break;
            case 9:
                lmask = _kshiftli_mask16(mask, 9);
                break;
            case 10:
                lmask = _kshiftli_mask16(mask, 10);
                break;
            case 11:
                lmask = _kshiftli_mask16(mask, 11);
                break;
            case 12:
                lmask = _kshiftli_mask16(mask, 12);
                break;
            case 13:
                lmask = _kshiftli_mask16(mask, 13);
                break;
            case 14:
                lmask = _kshiftli_mask16(mask, 14);
                break;
            case 15:
                lmask = _kshiftli_mask16(mask, 15);
        }

        switch(to) {
            case 0:
                rmask = _kshiftri_mask16(mask, 15);
                break;
            case 1:
                rmask = _kshiftri_mask16(mask, 14);
                break;
            case 2:
                rmask = _kshiftri_mask16(mask, 13);
                break;
            case 3:
                rmask = _kshiftri_mask16(mask, 12);
                break;
            case 4:
                rmask = _kshiftri_mask16(mask, 11);
                break;
            case 5:
                rmask = _kshiftri_mask16(mask, 10);
                break;
            case 6:
                rmask = _kshiftri_mask16(mask, 9);
                break;
            case 7:
                rmask = _kshiftri_mask16(mask, 8);
                break;
            case 8:
                rmask = _kshiftri_mask16(mask, 7);
                break;
            case 9:
                rmask = _kshiftri_mask16(mask, 6);
                break;
            case 10:
                rmask = _kshiftri_mask16(mask, 5);
                break;
            case 11:
                rmask = _kshiftri_mask16(mask, 4);
                break;
            case 12:
                rmask = _kshiftri_mask16(mask, 3);
                break;
            case 13:
                rmask = _kshiftri_mask16(mask, 2);
                break;
            case 14:
                rmask = _kshiftri_mask16(mask, 1);
                break;
            case 15:
                rmask = _kshiftri_mask16(mask, 0);
        }

        mask = _kand_mask16(lmask, rmask);     
    }

	return mask; 
}

__mmask8 _mm512_setmask_fromto_epi64(int from, int to)
{
    __mmask8 mask; 
    __mmask8 lmask; 
    __mmask8 rmask; 

	if (from > to || from > INT64_PER_M512_REG - 1 || to < 0) {
		mask = _mm512_movepi64_mask(_mm512_set1_epi64(0));
	} else {
		mask = _mm512_movepi64_mask(_mm512_set1_epi64(INT64_ALLBITS));

        if (from < 0 ) {
            from = 0;
        }

        if (to > INT64_PER_M512_REG - 1) {
            to = INT64_PER_M512_REG - 1;
        }

        switch (from) {
            case 0:
                lmask = _kshiftli_mask8(mask, 0);
                break;
            case 1:
                lmask = _kshiftli_mask8(mask, 1);
                break;
            case 2:
                lmask = _kshiftli_mask8(mask, 2);
                break;
            case 3:
                lmask = _kshiftli_mask8(mask, 3);
                break;
            case 4:
                lmask = _kshiftli_mask8(mask, 4);
                break;
            case 5:
                lmask = _kshiftli_mask8(mask, 5);
                break;
            case 6:
                lmask = _kshiftli_mask8(mask, 6);
                break;
            case 7:
                lmask = _kshiftli_mask8(mask, 7);
        }

        switch (to) {
            case 0:
                rmask = _kshiftri_mask8(mask, 7);
                break;
            case 1:
                rmask = _kshiftri_mask8(mask, 6);
                break;
            case 2:
                rmask = _kshiftri_mask8(mask, 5);
                break;
            case 3:
                rmask = _kshiftri_mask8(mask, 4);
                break;
            case 4:
                rmask = _kshiftri_mask8(mask, 3);
                break;
            case 5:
                rmask = _kshiftri_mask8(mask, 2);
                break;
            case 6:
                rmask = _kshiftri_mask8(mask, 1);
                break;
            case 7:
                rmask = _kshiftri_mask8(mask, 0);
        }

        mask = _kand_mask8(lmask, rmask);
    }

	return mask; 
}

__mmask16 _mm512_set_mask_epi32(int cutoff_index)
{
    return _mm512_setmask_fromto_epi32(0, cutoff_index);
}

__mmask8 _mm512_set_mask_epi64(int cutoff_index)
{
    return _mm512_setmask_fromto_epi64(0, cutoff_index);
}
#endif
