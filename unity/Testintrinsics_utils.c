#include "unity.h"
#include "mask_utils.h"
#include "intrinsics_utils.h"
#include <stdlib.h>
#include <float.h>

// Forward declarations needed to use Unity.
void setUp(void);
void tearDown(void);

// Forward declarations for initializing arrays and computing dot products.
void random_array(float *, int, float, float);
float serial_fdot(const float *, const float *, int);
float serial_fdot_indexed(const float *, const int *, const float *, int);
float serial_fdot_indexed2(const float *, const int *, const float *, const int *, int);
void test_dots(void);

int main(int argc, char *argv[])
{
    UNITY_BEGIN();

    RUN_TEST(test_dots);

    return UNITY_END();
}

void setUp(void) {}
void tearDown(void) {}

void random_array(float *x, int len, float a, float b)
{
    for (int i = 0; i < len; i++) {
        x[i] = a + (b - a) * ((float)rand() / RAND_MAX);
    }
}

float serial_fdot(const float *x, const float *y, int len)
{
    float sum = 0;

    for (int i = 0; i < len; i++) {
        sum += x[i] * y[i];
    }

    return sum;   
}

float serial_fdot_indexed(const float *x, const int *xindices, const float *y, int len)
{
    float sum = 0;
    int xind;

    for (int i = 0; i < len; i++) {
        xind = xindices[i];

        sum += x[xind] * y[i];
    }

    return sum;   
}

float serial_fdot_indexed2(const float *x, const int *xindices, const float *y, const int *yindices, int len)
{
    float sum = 0;
    int xind, yind;

    for (int i = 0; i < len; i++) {
        xind = xindices[i];
        yind = yindices[i];

        sum += x[xind] * y[yind];
    }

    return sum;   
}

void test_dots()
{
    int m = 100, n = 100;
    float *x = NULL, *y = NULL;
    const float fdelta = 1e-8;

    x = calloc(2*m, sizeof(float));

    TEST_ASSERT_NOT_NULL(x);

    if (x != NULL) {
        random_array(x, 2*m, -1, 1);
        y = x + m;

        float exact_dot = serial_fdot(x, y, m);
        float m256_dot = _mm256_fdot(x, y, m);
#ifdef SUPPORTS_AVX512
        float m512_dot = _mm512_fdot(x, y, m);
#endif
	    free(x);


        TEST_ASSERT_FLOAT_WITHIN(10 * FLT_EPSILON, exact_dot, m256_dot);
#ifdef SUPPORTS_AVX512
        TEST_ASSERT_FLOAT_WITHIN(10 * FLT_EPSILON, exact_dot, m512_dot);
#endif
    }
}
