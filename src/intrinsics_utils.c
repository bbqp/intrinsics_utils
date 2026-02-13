#include "mask_utils.h"
#include "intrinsics_utils.h"
#include <immintrin.h>
#include <stdio.h>

//----------------------------------------------------------------------------
// Functions for setting values of arrays.
//----------------------------------------------------------------------------

void _mm256_sset_value(float *x, int n, float value)
{
	int k;
	int cutoff = n % FLOAT_PER_M256_REG;
	__m256 vreg = _mm256_set1_ps(value);
	__m256i mask;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi32(cutoff);
		_mm256_maskstore_ps(x, mask, vreg);
	}

	for (k = cutoff; k < n; k += FLOAT_PER_M256_REG) {
		_mm256_store_ps(x + k, vreg);
	}
}

void _mm256_dset_value(double *x, int n, double value)
{
	int k;
	int cutoff = n % DOUBLE_PER_M256_REG;
	__m256d vreg = _mm256_set1_pd(value);
	__m256i mask;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi64(cutoff);
		_mm256_maskstore_pd(x, mask, vreg);
	}

	for (k = cutoff; k < n; k += DOUBLE_PER_M256_REG) {
		_mm256_store_pd(x + k, vreg);
	}
}

void _mm512_sset_value(float *x, int n, float value)
{
	int k;
	int cutoff = n % FLOAT_PER_M512_REG;
	__m512 vreg = _mm512_set1_ps(value);
	__mmask16 mask;

	if (cutoff > 0) {
		mask = _mm512_set_mask_epi32(cutoff);
		_mm256_mask_store_ps(x, mask, vreg);
	}

	for (k = cutoff; k < n; k += FLOAT_PER_M512_REG) {
		_mm512_store_ps(x + k, vreg);
	}
}



void _mm512_dset_value(double *x, int n, double value)
{
	int k;
	int cutoff = n % FLOAT_PER_M512_REG;
	__m512d vreg = _mm512_set1_ps(value);
	__mmask8 mask;

	if (cutoff > 0) {
		mask = _mm512_set_mask_epi64(cutoff);
		_mm512_mask_store_pd(x, mask, vreg);
	}

	for (k = cutoff; k < n; k += FLOAT_PER_M512_REG) {
		_mm512_store_pd(x + k, vreg);
	}
}

//----------------------------------------------------------------------------
// Functions for computing sums of elements in registers.
//----------------------------------------------------------------------------

float fdot_avx2(const float *x, const float *y, int n)
{
	__m256 xreg;
	__m256 yreg;
	__m256 preg;
	__m256 sreg = _mm256_set1_ps(0);
	__m256i mask;
	int i;
	int cutoff = n % FLOAT_PER_M256_REG;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi32(cutoff);
		
		xreg = _mm256_maskload_ps(x, mask);
		yreg = _mm256_maskload_ps(y, mask);
		preg = _mm256_mul_ps(xreg, yreg);
		sreg = _mm256_add_ps(sreg, preg);
	}

	for (i = cutoff; i < n; i += FLOAT_PER_M256_REG) {
		xreg = _mm256_load_ps(x + i);
		yreg = _mm256_load_ps(y + i);
		preg = _mm256_mul_ps(xreg, yreg);
		sreg = _mm256_add_ps(sreg, preg);
	}

	return _mm256_register_sum_ps(sreg);
}

float fdot_indexed_avx2(const float *x, const int *xindices, const float *y, int n)
{
	__m256 xreg;
	__m256 yreg;
	__m256 preg;
	__m256 sreg = _mm256_set1_ps(0);
	__m256i vindex;
	__m256i mask;
	int i;
	int cutoff = n % FLOAT_PER_M256_REG;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi32(cutoff);
		vindex = _mm256_maskload_epi32(xindices, mask);
		yreg = _mm256_maskload_ps(y, mask);
		xreg = _mm256_mask_i32gather_ps(sreg, x, vindex, _mm256_castsi256_ps(mask), 4);
		
		preg = _mm256_mul_ps(xreg, yreg);
		sreg = _mm256_add_ps(sreg, preg);
	}

	mask = _mm256_set_mask_epi32(INT32_PER_M256_REG);

	for (i = cutoff; i < n; i += FLOAT_PER_M256_REG) {
		vindex = _mm256_maskload_epi32(xindices + i, mask);
		yreg = _mm256_load_ps(y + i);
		xreg = _mm256_i32gather_ps(x, vindex, 4);

		preg = _mm256_mul_ps(xreg, yreg);
		sreg = _mm256_add_ps(sreg, preg);
	}

	return _mm256_register_sum_ps(sreg);
}

float fdot_indexed2_avx2(const float *x, const int *xindices, const float *y, const int *yindices, int n)
{
	__m256 xreg;
	__m256 yreg;
	__m256 preg;
	__m256 sreg = _mm256_set1_ps(0);
	__m256i xindex, yindex;
	__m256i mask;
	int i;
	int cutoff = n % FLOAT_PER_M256_REG;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi32(cutoff);
		xindex = _mm256_maskload_epi32(xindices, mask);
		yindex = _mm256_maskload_epi32(yindices, mask);

		xreg = _mm256_mask_i32gather_ps(sreg, x, xindex, _mm256_castsi256_ps(mask), 4);
		yreg = _mm256_mask_i32gather_ps(sreg, y, yindex, _mm256_castsi256_ps(mask), 4);
		
		preg = _mm256_mul_ps(xreg, yreg);
		sreg = _mm256_add_ps(sreg, preg);
	}

	mask = _mm256_set_mask_epi32(INT32_PER_M256_REG);

	for (i = cutoff; i < n; i += FLOAT_PER_M256_REG) {
		xindex = _mm256_maskload_epi32(xindices + i, mask);
		yindex = _mm256_maskload_epi32(yindices + i, mask);
		
		xreg = _mm256_i32gather_ps(x, xindex, 4);
		yreg = _mm256_i32gather_ps(y, yindex, 4);

		preg = _mm256_mul_ps(xreg, yreg);
		sreg = _mm256_add_ps(sreg, preg);
	}

	return _mm256_register_sum_ps(sreg);
}

double ddot_avx2(const double *x, const double *y, int n)
{
	__m256d xreg;
	__m256d yreg;
	__m256d preg;
	__m256d sreg = _mm256_set1_pd(0);
	__m256i mask;
	int i;
	int cutoff = n % DOUBLE_PER_M256_REG;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi64(cutoff);
		
		xreg = _mm256_maskload_pd(x, mask);
		yreg = _mm256_maskload_pd(y, mask);
		preg = _mm256_mul_pd(xreg, yreg);
		sreg = _mm256_add_pd(sreg, preg);
	}

	for (i = cutoff; i < n; i += DOUBLE_PER_M256_REG) {
		xreg = _mm256_load_pd(x + i);
		yreg = _mm256_load_pd(y + i);
		preg = _mm256_mul_pd(xreg, yreg);
		sreg = _mm256_add_pd(sreg, preg);
	}

	return _mm256_register_sum_pd(sreg);
}

double ddot_indexed_avx2(const double *x, const int *xindices, const double *y, int n)
{
	__m256d xreg;
	__m256d yreg;
	__m256d preg;
	__m256d sreg = _mm256_set1_pd(0);
	__m128i vindex;
	__m256i mask;
	__m256d dmask;
	__m128i mask128;
	int i;
	int cutoff = n % DOUBLE_PER_M256_REG;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi64(cutoff);
		dmask = _mm256_castsi256_pd(mask);
		mask128 = _mm_set_mask_epi32(cutoff);
		
		vindex = _mm_maskload_epi32(xindices, mask128);
		xreg = _mm256_mask_i32gather_pd(sreg, x, vindex, dmask, 8);
		yreg = _mm256_maskload_pd(y, mask);

		preg = _mm256_mul_pd(xreg, yreg);
		sreg = _mm256_add_pd(sreg, preg);
	}

	for (i = cutoff; i < n; i += DOUBLE_PER_M256_REG) {
		vindex = _mm_load_epi32(xindices + i);
		yreg = _mm256_load_pd(y + i);
		xreg = _mm256_i32gather_pd(x, vindex, 8);
		
		preg = _mm256_mul_pd(xreg, yreg);
		sreg = _mm256_add_pd(sreg, preg);
	}

	return _mm256_register_sum_pd(sreg);
}

double ddot_indexed2_avx2(const double *x, const int *xindices, const double *y, const int *yindices, int n)
{
	__m256d xreg;
	__m256d yreg;
	__m256d preg;
	__m256d sreg = _mm256_set1_pd(0);
	__m128i xindex, yindex;
	__m256i mask;
	__m256d dmask;
	__m128i mask128;
	int i;
	int cutoff = n % DOUBLE_PER_M256_REG;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi64(cutoff);
		dmask = _mm256_castsi256_pd(mask);
		mask128 = _mm_set_mask_epi32(cutoff);
		
		xindex = _mm_maskload_epi32(xindices, mask128);
		yindex = _mm_maskload_epi32(yindices, mask128);
		
		xreg = _mm256_mask_i32gather_pd(sreg, x, xindex, dmask, 8);
		yreg = _mm256_mask_i32gather_pd(sreg, y, yindex, dmask, 8);

		preg = _mm256_mul_pd(xreg, yreg);
		sreg = _mm256_add_pd(sreg, preg);
	}

	for (i = cutoff; i < n; i += DOUBLE_PER_M256_REG) {
		xindex = _mm_load_epi32(xindices + i);
		yindex = _mm_load_epi32(yindices + i);

		xreg = _mm256_i32gather_pd(x, xindex, 8);
		yreg = _mm256_i32gather_pd(y, yindex, 8);
		
		preg = _mm256_mul_pd(xreg, yreg);
		sreg = _mm256_add_pd(sreg, preg);
	}

	return _mm256_register_sum_pd(sreg);
}

float fdot_avx512(const float *, const float *, int)
{
	__m512 xreg;
	__m512 yreg;
	__m512 preg;
	__m512 sreg = _mm256_set1_ps(0);
	__mmask16 mask;
	int i;
	int cutoff = n % FLOAT_PER_M512_REG;

	if (cutoff > 0) {
		mask = _mm512_movepi32_mask(_mm512_set_mask_epi32(cutoff));	

		xreg = _mm512_maskz_load_ps(mask, x);
		yreg = _mm512_maskz_load_ps(mask, y);
		preg = _mm512_mul_ps(xreg, yreg);
		sreg = _mm512_add_ps(sreg, preg);
	}

	for (i = cutoff; i < n; i += FLOAT_PER_M256_REG) {
		xreg = _mm512_load_ps(x + i);
		yreg = _mm512_load_ps(y + i);
		preg = _mm512_mul_ps(xreg, yreg);
		sreg = _mm512_add_ps(sreg, preg);
	}

	return _mm512_register_sum_ps(sreg);
}


double ddot_avx512(const double *x, const double *y, int n)
{
    return 0;
}



float _mm_register_sum_ps(__m128 vreg)
{
    const int imm8 = 0xd8; // Swap positions 1 and 2: 0xd8 = 0b 1101 1000 = 0b 11 01 10 00

	vreg = _mm_hadd_ps(vreg, vreg);
    vreg = _mm_permute_ps(vreg, imm8);
    vreg = _mm_hadd_ps(vreg, vreg);
	
	vreg = _mm_hadd_ps(vreg, vreg);

	return _mm_cvtss_f32(vreg);
}

double _mm_register_sum_pd(__m128d vreg)
{
	vreg = _mm_hadd_pd(vreg, vreg);

	return _mm_cvtsd_f64(vreg);
}

<<<<<<< HEAD
=======

float _mm256_register_sum_ps(__m256 vreg)
{
	__m256i idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

	vreg = _mm256_hadd_ps(vreg, vreg);
	vreg = _mm256_permutevar8x32_ps(vreg, idx);
	vreg = _mm256_hadd_ps(vreg, vreg);
	vreg = _mm256_hadd_ps(vreg, vreg);

	return _mm256_cvtss_f32(vreg);
}

float _mm256_register_sum_pd(__m256d vreg)
{
	vreg = _mm256_hadd_pd(vreg, vreg);
	vreg = _mm256_hadd_pd(vreg, vreg);

	return _mm256_cvtsd_f64(vreg);
}

int _mm_count_nonzero_ps(__m128 a)
{
	int cmask = _mm_movemask_ps(a);

	return _popcnt32(cmask);
}

int _mm_count_nonzero_pd(__m128d a)
{
	int cmask = _mm_movemask_pd(a);

	return _popcnt32(cmask);
}

float _mm256_register_sum_ps(__m256 vreg)
{
	__m256i idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

	vreg = _mm256_hadd_ps(vreg, vreg);
	vreg = _mm256_permutevar8x32_ps(vreg, idx);
	vreg = _mm256_hadd_ps(vreg, vreg);
	vreg = _mm256_hadd_ps(vreg, vreg);

	return _mm256_cvtss_f32(vreg);
}

float _mm256_register_sum_pd(__m256d vreg)
{

    const int imm8 = 0xd8; // Swap positions 1 and 2: 0xd8 = 0b 1101 1000 = 0b 11 01 10 00

	vreg = _mm256_hadd_pd(vreg, vreg);
	vreg = _mm256_permute4x64_pd(vreg, imm8);
	vreg = _mm256_hadd_pd(vreg, vreg);
	
	return _mm256_cvtsd_f64(vreg);
}


int _mm256_count_nonzero_ps(__m256 a)
{
	int cmask = _mm256_movemask_ps(a);

	return _popcnt32(cmask);
}

int _mm256_count_nonzero_pd(__m256d a)
{
	int cmask = _mm256_movemask_pd(a);

	return _popcnt32(cmask);
}

float _mm512_register_sum_ps(__512 vreg)
{
    __m256 vlo = _mm512_extractf32x8_ps(vreg, 0);
    __m256 vhi = _mm512_extractf32x8_ps(vreg, 1);
	__m256i idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

	vlo = _mm256_hadd_ps(vlo, vlo);
	vhi = _mm256_hadd_ps(vhi, vhi);

	vlo = _mm256_permutevar8x32_ps(vlo, idx);
	vhi = _mm256_permutevar8x32_ps(vhi, idx);

    vlo = _mm256_hadd_ps(vlo, vlo);
    vhi = _mm256_hadd_ps(vhi, vhi);

    vlo = _mm256_hadd_ps(vlo, vlo);
    vhi = _mm256_hadd_ps(vhi, vhi); 

    return _mm256_cvtss_f32(vlo) + _mm256_cvtss_f32(vhi);
}

double _mm512_register_sum_pd(__m512d vreg)
{
    __m256 vlo = _mm512_extractf64x4_pd(vreg, 0);
    __m256 vhi = _mm512_extractf64x4_pd(vreg, 1);

    // Swap positions 1 and 2: 0xd8 = 0b 1101 1000 = 0b 11 01 10 00
    const int imm8 = 0xd8;

	vlo = _mm256_hadd_pd(vlo, vlo);
	vhi = _mm256_hadd_pd(vhi, vhi);

	vlo = _mm256_permute4x64_pd(vlo, imm8);
	vhi = _mm256_permute4x64_pd(vhi, imm8);

    vlo = _mm256_hadd_pd(vlo, vlo);
    vhi = _mm256_hadd_pd(vhi, vhi);

    return _mm256_cvtss_f32(vlo) + _mm256_cvtss_f32(vhi); 
}

int _mm512_count_nonzero_ps(__m512 vreg)
{
    __m256 vlo = _mm512_extractf32x8_ps(vreg, 0);
    __m256 vhi = _mm512_extractf32x8_ps(vreg, 1);
	int clo = _mm256_movemask_ps(clo);
	int chi = _mm256_movemask_ps(chi);

    return _popcnt32(clo) + _popcnt32(chi);
}

int _mm512_count_nonzero_pd(__m512d);
{
    __m256 vlo = _mm512_extractf64x4_pd(vreg, 0);
    __m256 vhi = _mm512_extractf64x4_pd(vreg, 1);
	int clo = _mm256_movemask_ps(clo);
	int chi = _mm256_movemask_ps(chi);

    return _popcnt32(clo) + _popcnt32(chi);
}

//----------------------------------------------------------------------------
// Functions for computing statistics of registers.
//----------------------------------------------------------------------------

float _mm_register_min_ps(__m128 a)
{
	int minidx = 0;
	float currmin = _mm_cvtss_f32(a);
	float nextmin;
	int i;

	nextmin = _mm_cvtss_f32(_mm_permute_ps(a, LPERM1));

	if (nextmin < currmin) {
		currmin = nextmin;
	}

	nextmin = _mm_cvtss_f32(_mm_permute_ps(a, LPERM2));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	nextmin = _mm_cvtss_f32(_mm_permute_ps(a, LPERM3));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	return currmin;
}

float _mm256_register_min_ps(__m256 a)
{	
	__m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

	int minidx = 0;
	float currmin = _mm256_cvtss_f32(a);
	float nextmin;
	int i;

	idx = _mm256_leftperm_epi32(idx, 1);
	nextmin = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(a, idx));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	idx = _mm256_leftperm_epi32(idx, 1);
	nextmin = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(a, idx));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}


	idx = _mm256_leftperm_epi32(idx, 1);
	nextmin = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(a, idx));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	idx = _mm256_leftperm_epi32(idx, 1);
	nextmin = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(a, idx));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	idx = _mm256_leftperm_epi32(idx, 1);
	nextmin = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(a, idx));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}


	idx = _mm256_leftperm_epi32(idx, 1);
	nextmin = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(a, idx));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	idx = _mm256_leftperm_epi32(idx, 1);
	nextmin = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(a, idx));
		
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	return currmin;
}

double _mm_register_min_pd(__m128d a)
{
	__m128d a0 = _mm_unpacklo_pd(a, a);
	__m128d a1 = _mm_unpackhi_pd(a, a);
	__m128d min = _mm_min_pd(a0, a1);

	return _mm_cvtsd_f64(min);
}

double _mm256_register_min_pd(__m256d a)
{
	const int imm8[3] = {
		0x39, // [0, 1, 2, 3] -> [1, 2, 3, 0] = 0b 00 11 10 01 = 0b(0011)(1001) = 0x39
		0x4e, // [0, 1, 2, 3] -> [2, 3, 0, 1] = 0b 01 00 11 10 = 0b(0100)(1110) = 0x4e
		0x93  // [0, 1, 2, 3] -> [3, 0, 1, 2] = 0b 10 01 00 11 = 0b(1001)(0011) = 0x93
	};

	int minidx = 0;
	double currmin = _mm256_cvtsd_f64(a);
	double nextmin;
	int i;

	nextmin = _mm256_cvtsd_f64(_mm256_permute4x64_pd(a, LPERM1));
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	nextmin = _mm256_cvtsd_f64(_mm256_permute4x64_pd(a, LPERM2));
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	nextmin = _mm256_cvtsd_f64(_mm256_permute4x64_pd(a, LPERM3));
	if (nextmin < currmin) {
		currmin = nextmin;
	}

	return currmin;
}

int _mm256_masksltnnz_epi64(__m256i a, __m256i mask)
{
	int displacement = 0;
	__m256i azero;

	if (!_mm256_testz_si256(a, a)) {
		// Get the zero positions in the remainder of the register a.
		azero = _mm256_andnot_si256(a, mask);
		
		while (_mm256_extract_epi64(azero, 0)) {
			azero = _mm256_leftshift_epi64(azero, 1);
			mask = _mm256_leftshift_epi64(mask, 1);
			
			displacement++;
		}
	}

	return displacement;
}

__m256d _mm256_leftperm_pd(__m256d a, int nperms)
{	
	switch (nperms) {
		case 1:
			_mm256_permute4x64_pd(a, M128_LPERM_TO_IMM8(1));
			break;
		case 2:
			_mm256_permute4x64_pd(a, M128_LPERM_TO_IMM8(2));
			break;
		case 3:
			_mm256_permute4x64_pd(a, M128_LPERM_TO_IMM8(3));
	}

	return a;
}

__m256d _mm256_rightperm_pd(__m256d a, int nperms)
{
	switch (nperms) {
		case 1:
			_mm256_permute4x64_pd(a, M128_RPERM_TO_IMM8(1));
			break;
		case 2:
			_mm256_permute4x64_pd(a, M128_RPERM_TO_IMM8(2));
			break;
		case 3:
			_mm256_permute4x64_pd(a, M128_RPERM_TO_IMM8(3));
	}

	return a;
}

__m256i _mm256_leftperm_epi64(__m256i a, int nperms)
{
	switch (nperms) {
		case 1:
			_mm256_permute4x64_epi64(a, M128_RPERM_TO_IMM8(1));
			break;
		case 2:
			_mm256_permute4x64_epi64(a, M128_RPERM_TO_IMM8(2));
			break;
		case 3:
			_mm256_permute4x64_epi64(a, M128_RPERM_TO_IMM8(3));
	}

	return a;
}

__m256i _mm256_maskleftperm_epi64(__m256i a, __m256i mask)
{
	return _mm256_and_si256(_mm256_leftperm_epi64(a, 1), mask);
}

__m256i _mm256_leftshift_epi64(__m256i a, int nshifts)
{
	__m256i mask;

	if (nshifts > 0) {
		mask = _mm256_set_mask_epi64(INT64_PER_M256_REG - nshifts);
		a = _mm256_maskleftperm_epi64(a, mask);
	}

	return a;
}

__m256i _mm256_leftperm_epi32(__m256i a, int nperms)
{
	__m256i lpidx = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
	int p;

	for (p = 0; p < nperms; p++) {
		a = _mm256_permutevar8x32_epi32(a, lpidx);
	}

	return a;
}

//----------------------------------------------------------------------------
// Helper routines for permuting
//----------------------------------------------------------------------------

__m128 _mm_shift_one_left_ps(__m128 a, int index)
{
	__m128i shift_indices;

	switch (index) {
		case 0:
			shift_indices = _mm_setr_epi32(3, 1, 2, 0);
			break;
		case 1:
			shift_indices = _mm_setr_epi32(1, 0, 2, 3);
			break;
		case 2:
			shift_indices = _mm_setr_epi32(0, 2, 1, 3);
			break;
		case 3:
			shift_indices = _mm_setr_epi32(0, 1, 3, 2);
	}

	return _mm_permutevar_ps(a, shift_indices);
}

__m128 _mm_shift_one_right_ps(__m128 a, int index)
{
	__m128i shift_indices;

	switch (index) {
		case 0:
			shift_indices = _mm_setr_epi32(1, 0, 2, 3);
			break;
		case 1:
			shift_indices = _mm_setr_epi32(0, 2, 1, 3);
			break;
		case 2:
			shift_indices = _mm_setr_epi32(0, 1, 3, 2);
			break;
		case 3:
			shift_indices = _mm_setr_epi32(3, 1, 2, 0);
	}

	return _mm_permutevar_ps(a, shift_indices);
}

//----------------------------------------------------------------------------
// Helper routines for copying data.
//----------------------------------------------------------------------------

void _mm256_copy1d_epi32(int *dst, const int *src, int n)
{
	int i;
	int cutoff = n % INT32_PER_M256_REG;
	__m256i mask;
	__m256i sreg, dreg;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi32(cutoff);
		
		sreg = _mm256_maskload_epi32(src, mask);
		_mm256_maskstore_epi32(dst, mask, sreg);
	}

	mask = _mm256_set1_epi32(INT32_ALLBITS);

	for (i = cutoff; i < n; i += INT32_PER_M256_REG) {
		sreg = _mm256_maskload_epi32(src, mask);
		_mm256_maskstore_epi32(dst, mask, sreg);
	}
}

void _mm256_copy2d_epi32(int *kind, int nrows, const int *iind, const int *jind, int numi, int numj)
{
	int i, j, jdidx, jsidx;
	int icutoff = numi % INT32_PER_M256_REG;
	int jcutoff = numj % INT32_PER_M256_REG;
	__m256i jreg, kreg;
	__m256i mask;

#ifdef CONTIGUOUS_LOOP
	for (j = 0; j < numj; j++) {
		jdidx = j * numi;
		jsidx = jind[j] * nrows;
		jreg = _mm256_set1_epi32(jsidx);
		
		if (icutoff > 0) {
			mask = _mm256_set_mask_epi32(icutoff);
			
			kreg = _mm256_maskload_epi32(iind, mask);
			kreg = _mm256_add_epi32(kreg, jreg);
			_mm256_maskstore_epi32(kind + jdidx, mask, kreg);
		}
		
		mask = _mm256_set1_epi32(INT32_PER_M256_REG);

		for (i = icutoff; i < numi; i += INT32_PER_M256_REG) {
			kreg = _mm256_maskload_epi32(iind + i, mask);
			kreg = _mm256_add_epi32(kreg, jreg);
			_mm256_maskstore_epi32(kind + jdidx + i, mask, kreg);
		}
	}
#else
	if (icutoff > 0) {
		mask = _mm256_set_mask_epi32(icutoff);

		for (j = 0; j < numj; j++) {
			jdidx = j * numi;
			jsidx = jind[j] * nrows;
			jreg = _mm256_set1_epi32(jsidx);
				
			kreg = _mm256_maskload_epi32(iind, mask);
			kreg = _mm256_add_epi32(kreg, jreg);
			_mm256_maskstore_epi32(kind + jdidx, mask, kreg);
		}
	}

	mask = _mm256_set1_epi32(INT32_ALLBITS);

	for (j = 0; j < numj; j++) {
		jdidx = j * numi;
		jsidx = jind[j] * nrows;
		jreg = _mm256_set1_epi32(jsidx);

		for (i = icutoff; i < numi; i += INT32_PER_M256_REG) {
			kreg = _mm256_maskload_epi32(iind + i, mask);
			kreg = _mm256_add_epi32(kreg, jreg);
			_mm256_maskstore_epi32(kind + jdidx + i, mask, kreg);
		}
	}
#endif
}


void _mm256_copy1d_ps(float *dst, const float *src, int n)
{
	int i;
	int cutoff = n % FLOAT_PER_M256_REG;
	__m256i mask;
	__m256 sreg, dreg;

	if (cutoff > 0) {
		mask = _mm256_set_mask_epi32(cutoff);
		
		sreg = _mm256_maskload_ps(src, mask);
		_mm256_maskstore_ps(dst, mask, sreg);
	}

	for (i = cutoff; i < n; i += INT32_PER_M256_REG) {
		sreg = _mm256_load_ps(src);
		_mm256_store_ps(dst, sreg);
	}
}

void _mm256_copy2d_indexed_ps(float *dst, const float *src, int nrows, const int *iind, const int *jind, int numi, int numj)
{
	int i, j, jdidx, jsidx;
	int icutoff = numi % FLOAT_PER_M256_REG;
	__m256i ireg, jreg, kreg;
	__m256i mask;
	__m256 sreg;
	__m256 zero = _mm256_set1_ps(0);

#ifdef CONTIGUOUS_LOOP
	for (j = 0; j < numj; j++) {
		jdidx = j * numi;
		jsidx = jind[j] * nrows;
		jreg = _mm256_set1_epi32(jsidx);
		
		if (icutoff > 0) {
			mask = _mm256_set_mask_epi32(icutoff);
			
			ireg = _mm256_maskload_epi32(iind, mask);
			kreg = _mm256_add_epi32(jreg, ireg);
			
			// Gather the data from the source.
			sreg = _mm256_mask_i32gather_ps(zero, src, kreg, _mm256_castsi256_ps(mask), 4);
			
			// Store the data.
			_mm256_maskstore_ps(dst + jdidx, mask, sreg);
		}
		
		mask = _mm256_set1_epi32(INT32_ALLBITS);

		for (i = icutoff; i < numi; i += INT32_PER_M256_REG) {
			ireg = _mm256_maskload_epi32(iind + i, mask);
			kreg = _mm256_add_epi32(jreg, ireg);
			
			// Gather the data from the source.
			sreg = _mm256_i32gather_ps(src, kreg, 4);
			
			// Store the data.
			_mm256_store_ps(dst + jdidx + i, sreg);
		}
	}
#else
	if (icutoff > 0) {
		mask = _mm256_set_mask_epi32(icutoff);

		for (j = 0; j < numj; j++) {
			jdidx = j * numi;
			jsidx = jind[j] * nrows;
			jreg = _mm256_set1_epi32(jsidx);
				
			ireg = _mm256_maskload_epi32(iind, mask);
			kreg = _mm256_add_epi32(jreg, ireg);
			sreg = _mm256_mask_i32gather_ps(zero, src, kreg, _mm256_castsi256_ps(mask), 4);
			_mm256_maskstore_ps(dst + jdidx, mask, sreg);
		}
	}

	mask = _mm256_set1_epi32(INT32_ALLBITS);

	for (j = 0; j < numj; j++) {
		jdidx = j * numi;
		jsidx = jind[j] * nrows;
		jreg = _mm256_set1_epi32(jsidx);

		for (i = icutoff; i < numi; i += INT32_PER_M256_REG) {
			ireg = _mm256_maskload_epi32(iind + i, mask);
			kreg = _mm256_add_epi32(jreg, ireg);
			sreg = _mm256_i32gather_ps(src, kreg, 4);
			_mm256_store_ps(dst + jdidx + i, sreg);
		}
	}
#endif
}

//----------------------------------------------------------------------------
// Helper routines for printing.
//----------------------------------------------------------------------------

void _mm256_print_register_epi32(__m256i a)
{
	int b[INT32_PER_M256_REG];
	__m256i mask = _mm256_set1_epi32(INT32_ALLBITS);

	_mm256_maskstore_epi32(b, mask, a);
	printf("%d %d %d %d %d %d %d %d\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
}

void _mm256_print_register_ps(__m256 a)
{
	float b[FLOAT_PER_M256_REG];

	_mm256_store_ps(b, a);
	printf("%g %g %g %g %g %g %g %g\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
}
