#include "unity.h"
#include "mask_utils.h"
#include "cpu_flags.h"
#include "constants.h"

#define NUM_RANGES (INT32_PER_M128_REG + 2)

static int m128_epi32_result[INT32_PER_M128_REG];
static int start_indices[NUM_RANGES];
static int end_indices[NUM_RANGES];

// Forward declarations needed to use Unity.
void setUp(void);
void tearDown(void);

// Forward declarations for testing register functions.
void test_mm_set_mask_fromto_epi32(void);
void test_mm_set_mask_epi32(void);

// Forward declarations for setting results for validation
void m128_epi32_set_result_fromto(int, int);
void m128_epi32_set_result_cutoff(int);

int main(int argc, char *argv[])
{
    UNITY_BEGIN();

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
    __m128i mask;
    int expected[INT32_PER_M128_REG];

    // Set the expected result.
    m128_epi32_set_result_cutoff(cutoff);

    // Store the expected result.
    mask = _mm_set_mask_epi32(cutoff);
    _mm_store_epi32(mask, expected);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, m128_epi32_result, INT32_PER_M128_REG);
}

void m128_epi32_set_result_fromto(int from, int to)
{
    if (from > to) {
        for (int i = 0; i < INT32_PER_M128_REG; i++) {
            m128_epi32_result[i] = 0;
        }
    } else {
        if (from < 0) {
            from = 0;
        }

        if (to > INT32_PER_M128_REG - 1) {
            to = INT32_PER_M128_REG - 1;
        }

        for (int i = from; i <= to; i++) {
            m128_epi32_result[i] = INT32_ALLBITS;
        }
    }
}

void m128_epi32_set_result_cutoff(int cutoff)
{
    m128_epi32_set_result_fromto(0, cutoff);
}
