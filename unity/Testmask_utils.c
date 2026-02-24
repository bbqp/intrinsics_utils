#include "unity.h"
#include "mask_utils.h"
#include "cpu_flags.h"
#include "constants.h"
#include <immintrin.h>

#ifdef SUPPORTS_AVX512
#define MAX_BUFFER_SIZE INT32_PER_M512_REG
#else
#define MAX_BUFFER_SIZE INT32_PER_M256_REG
#endif

#define M128_INDICES_LEN (INT32_PER_M128_REG + 2)
#define M256_INDICES_LEN (INT32_PER_M256_REG + 2)
#define M512_INDICES_LEN (INT32_PER_M512_REG + 2)

#define M128_64BIT_INDICES_LEN (INT64_PER_M128_REG + 2)
#define M256_64BIT_INDICES_LEN (INT64_PER_M256_REG + 2)
#define M512_64BIT_INDICES_LEN (INT64_PER_M512_REG + 2)

// This is a little annoying, but using int64_t, __int64_t, and int_least64_t
// (aliases for long int) are insufficient for the arguments needed for
// certain integer-related intrinsics which require a pointer to type __int64,
// which itself is an alias for long long int. Overall, this has been a weird
// quirk to have to work around. It could be an AMD Ryzen issue, but I may
// keep this around for testing as needed.
typedef long long int int_atleast64_t;

// Arrays used for testing functions.
static int32_t epi32_expected[MAX_BUFFER_SIZE];
static int32_t epi32_actual[MAX_BUFFER_SIZE];

static int_atleast64_t epi64_expected[MAX_BUFFER_SIZE];
static int_atleast64_t epi64_actual[MAX_BUFFER_SIZE];

static float ps_expected[MAX_BUFFER_SIZE];
static float ps_actual[MAX_BUFFER_SIZE];

static double pd_expected[MAX_BUFFER_SIZE];
static double pd_actual[MAX_BUFFER_SIZE];

// Arrays of indices for performing tests.
static int m128_indices[M128_INDICES_LEN];
static int m256_indices[M256_INDICES_LEN];
static int m512_indices[M512_INDICES_LEN];

// Forward declarations needed to use Unity.
void setUp(void);
void tearDown(void);

// Functions for 
void test_mm_set_mask_fromto_epi32(void);
void test_mm_set_mask_epi32(void);
void test_mm_set_mask_fromto_epi64(void);
void test_mm_set_mask_epi64(void);

void test_mm256_set_mask_fromto_epi32(void);
void test_mm256_set_mask_epi32(void);

void test_mm512_set_mask_fromto_epi32(void);
void test_mm512_set_mask_epi32(void);

// Functions for setting expected mask results in XMM registers.
void m128_epi32_set_expected_fromto(int, int);
void m128_epi32_set_expected_to(int);
void m128_epi64_set_expected_fromto(int, int);
void m128_epi64_set_expected_to(int);
void m128_ps_set_expected_to(int);
void m128_pd_set_expected_to(int);

void m256_epi32_set_expected_fromto(int, int);
void m256_epi32_set_expected_to(int);
void m256_epi64_set_expected_fromto(int, int);
void m256_epi64_set_expected_to(int);
void m256_ps_set_expected_to(int);
void m256_pd_set_expected_to(int);

#ifdef SUPPORTS_AVX512
void m512_epi32_set_expected_fromto(int, int);
void m512_epi32_set_expected_to(int);
void m512_epi64_set_expected_fromto(int, int);
void m512_epi64_set_expected_to(int);
#endif

int main(int argc, char *argv[])
{
    UNITY_BEGIN();

    RUN_TEST(test_mm_set_mask_fromto_epi32);
    RUN_TEST(test_mm_set_mask_epi32);
    RUN_TEST(test_mm_set_mask_epi64);

    RUN_TEST(test_mm256_set_mask_fromto_epi32);
    RUN_TEST(test_mm256_set_mask_epi32);

    RUN_TEST(test_mm512_set_mask_fromto_epi32);
    RUN_TEST(test_mm512_set_mask_epi32);

    return UNITY_END();
}

void setUp(void)
{
    for (int i = -1; i < M128_INDICES_LEN + 1; i++) {
        m128_indices[i + 1] = i;
    }

    for (int i = -1; i < M256_INDICES_LEN + 1; i++) {
        m256_indices[i + 1] = i;
    }

#ifdef SUPPORTS_AVX512
    for (int i = -1; i < M512_INDICES_LEN + 1; i++) {
        m512_indices[i + 1] = i;
    }
#endif
}

void tearDown(void)
{
    // Nothing to do here since nothing is allocated on the heap.
}

void test_mm_set_mask_fromto_epi32(void)
{
    int start, end;
    __m128i result_mask;
    __m128i store_mask = _mm_set1_epi32(INT32_ALLBITS);

    // Test results where the ending index is greater than or equal to the starting index.
    for (int i = 0; i < M128_INDICES_LEN; i++) {
        start = m128_indices[i];

        for (int j = i; j < M128_INDICES_LEN; j++) {
            end = m128_indices[j];

            // Set the expected result.
            m128_epi32_set_expected_fromto(start, end);

            // Store the actual result.
            result_mask = _mm_setmask_fromto_epi32(start, end);
            _mm_maskstore_epi32(epi32_actual, store_mask, result_mask);

            TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M128_REG);
        }
    }

    // Test results where the ending index is less than the starting index.
    for (int i = 0; i < M128_INDICES_LEN; i++) {
        start = m128_indices[i];

        for (int j = i - 1; j >= 0; j--) {
            end = m128_indices[j];

            // Set the expected result.
            m128_epi32_set_expected_fromto(start, end);

            // Store the actual result.
            result_mask = _mm_setmask_fromto_epi32(start, end);
            _mm_maskstore_epi32(epi32_actual, store_mask, result_mask);

            TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M128_REG);
        }
    }
}

void test_mm_set_mask_epi32(void)
{
    int cutoff_index;
    __m128i result_mask;
    __m128i store_mask = _mm_set1_epi32(INT32_ALLBITS);

    for (int i = 0; i < M128_INDICES_LEN; i++) {
        // Set the index of the last nonzero element in the mask.
        cutoff_index = m128_indices[i];

        // Set the expected result.
        m128_epi32_set_expected_to(cutoff_index);

        // Store the actual result.
        result_mask = _mm_set_mask_epi32(cutoff_index);
        _mm_maskstore_epi32(epi32_actual, store_mask, result_mask);

        TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M128_REG);
    }
}

void test_mm_set_mask_fromto_epi64(void)
{
    TEST_IGNORE();
}

void test_mm_set_mask_epi64(void)
{
    int cutoff_index;
    __m128i result_mask;
    __m128i store_mask = _mm_set1_epi64x(INT64_ALLBITS);

    for (int i = 0; i < M128_64BIT_INDICES_LEN; i++) {
        // Set the index of the last nonzero element in the mask.
        cutoff_index = m128_indices[i];

        // Set the expected result.
        m128_epi64_set_expected_to(cutoff_index);

        // Store the actual result.
        result_mask = _mm_set_mask_epi64(cutoff_index);
        _mm_maskstore_epi64(epi64_actual, store_mask, result_mask);

        TEST_ASSERT_EQUAL_INT64_ARRAY(epi64_expected, epi64_actual, INT64_PER_M128_REG);
    }
}

void test_mm256_set_mask_fromto_epi32(void)
{
    int start, end;
    __m256i result_mask;
    __m256i store_mask = _mm256_set1_epi32(INT32_ALLBITS);

    // Test results where the ending index is greater than or equal to the starting index.
    for (int i = 0; i < M256_INDICES_LEN; i++) {
        start = m256_indices[i];

        for (int j = i; j < M256_INDICES_LEN; j++) {
            end = m256_indices[j];

            // Set the expected result.
            m256_epi32_set_expected_fromto(start, end);

            // Store the actual result.
            result_mask = _mm256_setmask_fromto_epi32(start, end);
            _mm256_maskstore_epi32(epi32_actual, store_mask, result_mask);

            TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M256_REG);
        }
    }

    // Test results where the ending index is less than the starting index.
    for (int i = 0; i < M256_INDICES_LEN; i++) {
        start = m256_indices[i];

        for (int j = i - 1; j >= 0; j--) {
            end = m256_indices[j];

            // Set the expected result.
            m256_epi32_set_expected_fromto(start, end);

            // Store the actual result.
            result_mask = _mm256_setmask_fromto_epi32(start, end);
            _mm256_maskstore_epi32(epi32_actual, store_mask, result_mask);

            TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M256_REG);
        }
    }
}

void test_mm256_set_mask_epi32(void)
{
    int cutoff_index;
    __m256i result_mask;
    __m256i store_mask = _mm256_set1_epi32(INT32_ALLBITS);

    for (int i = 0; i < M256_INDICES_LEN; i++) {
        // Set the index of the last nonzero element in the mask.
        cutoff_index = m256_indices[i];

        // Set the expected result.
        m256_epi32_set_expected_to(cutoff_index);

        // Store the actual result.
        result_mask = _mm256_set_mask_epi32(cutoff_index);
        _mm256_maskstore_epi32(epi32_actual, store_mask, result_mask);

        TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M256_REG);
    }
}

void test_mm512_set_mask_fromto_epi32(void)
{
    int start, end;
    __mmask16 result_mask;

    // Test results where the ending index is greater than or equal to the starting index.
    for (int i = 0; i < M512_INDICES_LEN; i++) {
        start = m512_indices[i];

        for (int j = i; j < M512_INDICES_LEN; j++) {
            end = m512_indices[j];

            // Set the expected result.
            m512_epi32_set_expected_fromto(start, end);

            // Store the actual result.
            result_mask = _mm512_setmask_fromto_epi32(start, end);
            _mm512_storeu_epi32(epi32_actual, _mm512_movm_epi32(result_mask));

            TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M512_REG);
        }
    }

    // Test results where the ending index is less than the starting index.
    for (int i = 0; i < M512_INDICES_LEN; i++) {
        start = m512_indices[i];

        for (int j = i - 1; j >= 0; j--) {
            end = m512_indices[j];

            // Set the expected result.
            m512_epi32_set_expected_fromto(start, end);

            // Store the actual result.
            result_mask = _mm512_setmask_fromto_epi32(start, end);
            _mm512_storeu_epi32(epi32_actual, _mm512_movm_epi32(result_mask));

            TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M512_REG);
        }
    }
}

void test_mm512_set_mask_epi32(void)
{
    int cutoff_index;
    __mmask16 result_mask;

    for (int i = 3; i < M512_INDICES_LEN; i++) {
        // Set the index of the last nonzero element in the mask.
        cutoff_index = m512_indices[i];

        // Set the expected result.
        m512_epi32_set_expected_to(cutoff_index);

        // Store the actual result.
        result_mask = _mm512_set_mask_epi32(cutoff_index);
        _mm512_storeu_epi32(epi32_actual, _mm512_movm_epi32(result_mask));

        TEST_ASSERT_EQUAL_INT32_ARRAY(epi32_expected, epi32_actual, INT32_PER_M512_REG);
    }
}

//----------------------------------------------------------------------------
// Functions for setting expected mask results in XMM registers.
//----------------------------------------------------------------------------

void m128_epi32_set_expected_fromto(int from, int to)
{
    if (from > to || from > INT32_PER_M128_REG - 1 || to < 0) {
        for (int i = 0; i < INT32_PER_M128_REG; i++) {
            epi32_expected[i] = 0;
        }
    } else {
        for (int i = 0; i < INT32_PER_M128_REG; i++) {
            epi32_expected[i] = 0;
        }
 
        if (from < 0) {
            from = 0;
        }

        if (to > INT32_PER_M128_REG - 1) {
            to = INT32_PER_M128_REG - 1;
        }

        for (int i = from; i <= to; i++) {
            epi32_expected[i] = INT32_ALLBITS;
        }
    }
}

void m128_epi32_set_expected_to(int cutoff_index)
{
    m128_epi32_set_expected_fromto(0, cutoff_index);
}

void m128_epi64_set_expected_fromto(int from, int to)
{
    if (from > to || from > INT64_PER_M128_REG || to < 0) {
        for (int i = 0; i < INT64_PER_M128_REG; i++) {
            epi64_expected[i] = 0;
        }
    } else {
        for (int i = 0; i < INT64_PER_M128_REG; i++) {
            epi64_expected[i] = 0;
        }
 
        if (from < 0) {
            from = 0;
        }

        if (to > INT64_PER_M128_REG - 1) {
            to = INT64_PER_M128_REG - 1;
        }

        for (int i = from; i <= to; i++) {
            epi64_expected[i] = INT64_ALLBITS;
        }
    }
}

void m128_epi64_set_expected_to(int cutoff_index)
{
    m128_epi64_set_expected_fromto(0, cutoff_index);
}

void m128_ps_set_expected_to(int cutoff_index)
{
    m128_epi32_set_expected_to(cutoff_index);
    __m128i lmask = _mm_set1_epi32(INT32_ALLBITS);
    __m128i regi = _mm_maskload_epi32(epi32_expected, lmask);
    __m128 regs = _mm_castsi128_ps(regi);
    _mm_store_ps(ps_expected, regs);
}

void m128_pd_set_expected_to(int cutoff_index)
{
    m128_epi64_set_expected_to(cutoff_index);
    __m128i lmask = _mm_set1_epi64x(INT64_ALLBITS);
    __m128i regi = _mm_maskload_epi64(epi64_expected, lmask);
    __m128d regs = _mm_castsi128_pd(regi);
    _mm_store_pd(pd_expected, regs);
}

//----------------------------------------------------------------------------
// Functions for setting expected mask results in YMM registers.
//----------------------------------------------------------------------------

void m256_epi32_set_expected_fromto(int from, int to)
{
    if (from > to || from > INT32_PER_M256_REG - 1 || to < 0) {
        for (int i = 0; i < INT32_PER_M256_REG; i++) {
            epi32_expected[i] = 0;
        }
    } else {
        for (int i = 0; i < INT32_PER_M256_REG; i++) {
            epi32_expected[i] = 0;
        }
 
        if (from < 0) {
            from = 0;
        }

        if (to > INT32_PER_M256_REG - 1) {
            to = INT32_PER_M256_REG - 1;
        }

        for (int i = from; i <= to; i++) {
            epi32_expected[i] = INT32_ALLBITS;
        }
    }
}

void m256_epi32_set_expected_to(int cutoff_index)
{
    m256_epi32_set_expected_fromto(0, cutoff_index);
}

void m256_epi64_set_expected_fromto(int from, int to)
{
    if (from > to || from > INT64_PER_M256_REG - 1 || to < 0) {
        for (int i = 0; i < INT64_PER_M256_REG; i++) {
            epi64_expected[i] = 0;
        }
    } else {
        for (int i = 0; i < INT64_PER_M256_REG; i++) {
            epi64_expected[i] = 0;
        }
 
        if (from < 0) {
            from = 0;
        }

        if (to > INT64_PER_M256_REG - 1) {
            to = INT64_PER_M256_REG - 1;
        }

        for (int i = from; i <= to; i++) {
            epi64_expected[i] = INT64_ALLBITS;
        }
    }
}

void m256_epi64_set_expected_to(int cutoff_index)
{
    m256_epi64_set_expected_fromto(0, cutoff_index);
}

void m256_ps_set_expected_to(int cutoff_index)
{
    m256_epi32_set_expected_to(cutoff_index);
    __m256i lmask = _mm256_set1_epi32(INT32_ALLBITS);
    __m256i regi = _mm256_maskload_epi32(epi32_expected, lmask);
    __m256 regs = _mm256_castsi256_ps(regi);
    _mm256_store_ps(ps_expected, regs);
}

void m256_pd_set_expected_to(int cutoff_index)
{
    m256_epi64_set_expected_to(cutoff_index);
    __m256i lmask = _mm256_set1_epi64x(INT64_ALLBITS);
    __m256i regi = _mm256_maskload_epi64(epi64_expected, lmask);
    __m256d regs = _mm256_castsi256_pd(regi);
    _mm256_store_pd(pd_expected, regs);
}

#ifdef SUPPORTS_AVX512
void m512_epi32_set_expected_fromto(int from, int to)
{
    if (from > to || from > INT32_PER_M512_REG - 1 || to < 0) {
        for (int i = 0; i < INT32_PER_M512_REG; i++) {
            epi32_expected[i] = 0;
        }
    } else {
        for (int i = 0; i < INT32_PER_M512_REG; i++) {
            epi32_expected[i] = 0;
        }
 
        if (from < 0) {
            from = 0;
        }

        if (to > INT32_PER_M512_REG - 1) {
            to = INT32_PER_M512_REG - 1;
        }

        for (int i = from; i <= to; i++) {
            epi32_expected[i] = INT32_ALLBITS;
        }
    }
}

void m512_epi32_set_expected_to(int cutoff_index)
{
    m512_epi32_set_expected_fromto(0, cutoff_index);
}

void m512_epi64_set_expected_fromto(int from, int to)
{
    if (from > to || from > INT64_PER_M512_REG - 1 || to < 0) {
        for (int i = 0; i < INT64_PER_M512_REG; i++) {
            epi64_expected[i] = 0;
        }
    } else {
        for (int i = 0; i < INT64_PER_M512_REG; i++) {
            epi64_expected[i] = 0;
        }
 
        if (from < 0) {
            from = 0;
        }

        if (to > INT64_PER_M512_REG - 1) {
            to = INT64_PER_M512_REG - 1;
        }

        for (int i = from; i <= to; i++) {
            epi64_expected[i] = INT64_ALLBITS;
        }
    }
}

void m512_epi64_set_expected_to(int cutoff_index)
{
    m512_epi64_set_expected_fromto(0, cutoff_index);
}
#endif 
