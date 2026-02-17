#include "unity.h"
#include "mask_utils.h"
#include "cpu_flags.h"
#include "constants.h"
#include <immintrin.h>

#define NUM_RANGES (INT32_PER_M128_REG + 2)

static int m128_epi32_expected[INT32_PER_M128_REG];
static int m128_epi32_actual[INT32_PER_M128_REG];
static int start_indices[NUM_RANGES];
static int end_indices[NUM_RANGES];

// Forward declarations needed to use Unity.
void setUp(void);
void tearDown(void);

// Forward declarations for testing register functions.
void test_mm_set_mask_fromto_epi32(void);
void test_mm_set_mask_epi32(void);

// Forward declarations for setting results for validation
void m128_epi32_set_expected_fromto(int, int);
void m128_epi32_set_expected_cutoff(int);

int main(int argc, char *argv[])
{
    UNITY_BEGIN();

    RUN_TEST(test_mm_set_mask_fromto_epi32);
    RUN_TEST(test_mm_set_mask_epi32);

    return UNITY_END();
}

void setUp(void)
{
    for (int i = 0; i < NUM_RANGES; i++) {
        start_indices[i] = i - 1;
        end_indices[i] = i - 1;
    }
}

void tearDown(void) {}


void test_mm_set_mask_fromto_epi32(void)
{
    TEST_PASS();
}

void test_mm_set_mask_epi32(void)
{
    int cutoff = INT32_PER_M128_REG / 2 - 1;
    __m128i result_mask;
    __m128i store_mask = _mm_set1_epi32(INT32_ALLBITS);

    for (int i = 0; i < NUM_RANGES; i++) {
        // Set the index of the last nonzero element in the mask.
        cutoff = end_indices[i];

        // Set the expected result.
        m128_epi32_set_expected_cutoff(cutoff);

        // Store the actual result.
        result_mask = _mm_set_mask_epi32(cutoff);
        _mm_maskstore_epi32(m128_epi32_actual, store_mask, result_mask);

        TEST_ASSERT_EQUAL_INT32_ARRAY(m128_epi32_expected, m128_epi32_actual, INT32_PER_M128_REG);
    }
}

void m128_epi32_set_expected_fromto(int from, int to)
{
    if (from > to) {
        for (int i = 0; i < INT32_PER_M128_REG; i++) {
            m128_epi32_expected[i] = 0;
        }
    } else {
        if (from < 0) {
            from = 0;
        }

        if (to > INT32_PER_M128_REG - 1) {
            to = INT32_PER_M128_REG - 1;
        }

        for (int i = from; i <= to; i++) {
            m128_epi32_expected[i] = INT32_ALLBITS;
        }
    }
}

void m128_epi32_set_expected_cutoff(int cutoff)
{
    m128_epi32_set_expected_fromto(0, cutoff);
}
