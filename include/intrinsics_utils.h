#ifndef INTRINSICS_UTILS
#define INTRINSICS_UTILS

#include <immintrin.h>
#include <stdint.h>

// Macros for imm8 constants for permuting/shuffling.
#define INT8_ALLBITS ((int32_t)0xff)

#define LPERM0 0xe4
#define LPERM1 0x39
#define LPERM2 0x4e
#define LPERM3 0x93

#define M128_LPERM_MASK (LPERM0 + (LPERM1 << 8) + (LPERM2 << 16) + (LPERM3 << 24))
#define M128_LPERM_TO_IMM8(NPERMS) (((M128_LPERM_MASK) >> ((NPERMS) * 8)) & INT8_ALLBITS)

#define RPERM0 LPERM0
#define RPERM1 LPERM3
#define RPERM2 LPERM2
#define RPERM3 LPERM1

#define M128_RPERM_MASK (RPERM0 + ((RPERM1) << 8) + ((RPERM2) << 16) + ((RPERM3) << 24))
#define M128_RPERM_TO_IMM8(NPERMS) (((M128_RPERM_MASK) >> ((NPERMS) * 8)) & INT8_ALLBITS)


//----------------------------------------------------------------------------
// Functions for setting values of arrays.
//----------------------------------------------------------------------------

void _mm256_sset_value(float *, int, float);
void _mm256_dset_value(double *, int, double);

void _mm512_sset_value(float *, int, float);
void _mm512_dset_value(double *, int, double);

//----------------------------------------------------------------------------
// Functions for computing sums of elements in registers.
//----------------------------------------------------------------------------

float _mm256_fdot(const float *, const float *, int);
float _mm256_fdot_indexed(const float *, const int *, const float *, int);
float _mm256_fdot_indexed2(const float *, const int *, const float *, const int *, int);

double _mm256_ddot(const double *, const double *, int);
double _mm256_ddot_indexed(const double *, const int *, const double *, int);
double _mm256_ddot_indexed2(const double *, const int *, const double *, const int *, int);

float _mm512_fdot(const float *, const float *, int);
float _mm512_fdot_indexed(const float *, const int *, const float *, int);
float _mm512_fdot_indexed2(const float *, const int *, const float *, const int *, int);

double _mm512_ddot(const double *, const double *, int);

float _mm_register_sum_ps(__m128);
double _mm_register_sum_pd(__m128d);
int _mm_count_nonzero_ps(__m128);
int _mm_count_nonzero_pd(__m128d);

float _mm256_register_sum_ps(__m256);
double _mm256_register_sum_pd(__m256d);
int _mm256_count_nonzero_ps(__m256);
int _mm256_count_nonzero_pd(__m256d);

float _mm512_register_sum_ps(__512);
double _mm512_register_sum_pd(__m512d);
int _mm512_count_nonzero_ps(__m512);
int _mm512_count_nonzero_pd(__m512d);

//----------------------------------------------------------------------------
// Functions for computing statistics of registers.
//----------------------------------------------------------------------------

float _mm_register_min_ps(__m128);
float _mm256_register_min_ps(__m256);
double _mm_register_min_pd(__m128d);
double _mm256_register_min_pd(__m256d);


//----------------------------------------------------------------------------
// Functions for permuting elements in registers.
//----------------------------------------------------------------------------

__m256d _mm256_leftperm_pd(__m256d, int);
__m256d _mm256_rightperm_pd(__m256d, int);
__m256i _mm256_leftperm_epi64(__m256i, int);
__m256i _mm256_leftperm_epi32(__m256i, int);

//----------------------------------------------------------------------------
// Helper routines for permuting
//----------------------------------------------------------------------------

__m128 _mm_shift_one_left_ps(__m128, int);
__m128 _mm_shift_one_right_ps(__m128, int);

//----------------------------------------------------------------------------
// Helper routines for copying data.
//----------------------------------------------------------------------------

void _mm256_copy1d_epi32(int *, const int *, int);
void _mm256_copy2d_epi32(int *, int, const int *, const int *, int, int);

void _mm256_copy1d_ps(float *, const float *, int);
void _mm256_copy2d_indexed_ps(float *, const float *, int, const int *, const int *, int, int);

//----------------------------------------------------------------------------
// Helper routines for printing.
//----------------------------------------------------------------------------

void _mm256_print_register_epi32(__m256i);
void _mm256_print_register_ps(__m256);

#endif
