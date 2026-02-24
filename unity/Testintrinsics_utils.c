#include "unity.h"
#include "mask_utils.h"
#include "intrinsics_utils.h"
#include <stdlib.h>
#include <float.h>

// Forward declarations needed to use Unity.
void setUp(void);
void tearDown(void);

// Forward declarations for initializing arrays and computing dot products.
void random_farray(float *, int, float, float);
void set_farray(float *, int, float);
void copy_farray(float *, const float *, int);
float serial_fdot(const float *, const float *, int);
float serial_fsum(const float *, int);
float serial_fdot_kahan(const float *, const float *, int);

// Forward declarations for double precision functions.
void random_darray(double *, int, double, double);
void set_darray(double *, int, double);
void copy_darray(double *, const double *, int);
double serial_ddot(const double *, const double *, int);
double serial_dsum(const double *, int);
double serial_ddot_kahan(const double *, const double *, int);

// Forward declarations for tests.
void test_fdots(void);
void test_ddots(void);

int main(int argc, char *argv[])
{
    UNITY_BEGIN();

    RUN_TEST(test_fdots);
    RUN_TEST(test_ddots);

    return UNITY_END();
}

void setUp(void)
{
    srand(0);
}

void tearDown(void) {}

void random_farray(float *x, int len, float a, float b)
{
    for (int i = 0; i < len; i++) {
        x[i] = a + (b - a) * ((float)rand() / RAND_MAX);
    }
}

void set_farray(float *x, int len, float value)
{
    for (int i = 0; i < len; i++) {
        x[i] = value;
    }
}

void copy_farray(float *dst, const float *src, int len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

float serial_fdot_kahan(const float *x, const float *y, int len)
{
    float z, t;
    float p, sum = 0, c = 0;

    for (int i = 0; i < len; i++) {
        z = x[i] * y[i] - c;
        t = sum + z;
        c = (t - sum) - z;
        sum = t;
    }

    return sum;
}

float serial_fdot(const float *x, const float *y, int len)
{
    float sum = 0;

    for (int i = 0; i < len; i++) {
        sum += x[i] * y[i];
    }

    return sum;   
}

float serial_fsum(const float *x, int len)
{
    float sum = 0;

    for (int i = 0; i < len; i++) {
        sum += x[i];
    }

    return sum;   
}

void random_darray(double *x, int len, double a, double b)
{
    for (int i = 0; i < len; i++) {
        x[i] = a + (b - a) * ((double)rand() / RAND_MAX);
    }
}

void set_darray(double *x, int len, double value)
{
    for (int i = 0; i < len; i++) {
        x[i] = value;
    }
}

void copy_darray(double *dst, const double *src, int len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

double serial_ddot(const double *x, const double *y, int len)
{
    double sum = 0;

    for (int i = 0; i < len; i++) {
        sum += x[i] * y[i];
    }

    return sum;   
}

double serial_dsum(const double *x, int len)
{
    double sum = 0;

    for (int i = 0; i < len; i++) {
        sum += x[i];
    }

    return sum;   
}

double serial_ddot_kahan(const double *x, const double *y, int len)
{
    double z, t;
    double p, sum = 0, c = 0;

    for (int i = 0; i < len; i++) {
        z = x[i] * y[i] - c;
        t = sum + z;
        c = (t - sum) - z;
        sum = t;
    }

    return sum;
}

//----------------------------------------------------------------------------
// Tests for intrinsics.
//----------------------------------------------------------------------------

void test_fdots()
{
    int m = 15, n = 100;
    float *x = NULL, *y = NULL, *accumulator = NULL;

    x = calloc(2*m, sizeof(float));

    TEST_ASSERT_NOT_NULL(x);

    if (x != NULL) {
        y = x + m;

        set_farray(x, m, 0.1f);
        set_farray(y, m, 1.0f);

        float delta;
        float element = x[0];
        float exact_dot = (float)m / 10;
        float serial_dot = serial_fdot(x, y, m);
        float serial_add = serial_fsum(x, m);
        float serial_dot_kahan = serial_fdot_kahan(x, y, m);
        float m256_dot = _mm256_fdot(x, y, m);
#ifdef SUPPORTS_AVX512
        float m512_dot = _mm512_fdot(x, y, m);
#endif
	    free(x);

        delta = 10 * exact_dot * FLT_EPSILON;
        if (delta < 0) delta = -delta;

        printf("element:       %f\n", element);
        printf("exact:         %f\n", exact_dot);
        printf("serial:        %f\n", serial_dot);
        printf("serial sum:    %f\n", serial_add);
        printf("serial kahan:  %f\n", serial_dot_kahan);
        printf("m256:          %f\n", m256_dot);

#ifdef SUPPORTS_AVX512
        printf("m512:         %f\n", m512_dot);
#endif

        TEST_ASSERT_FLOAT_WITHIN(delta, exact_dot, serial_dot);
        TEST_ASSERT_FLOAT_WITHIN(delta, exact_dot, serial_dot_kahan);
        TEST_ASSERT_FLOAT_WITHIN(delta, exact_dot, m256_dot);
        TEST_ASSERT_FLOAT_WITHIN(delta, serial_dot, serial_dot_kahan);
        TEST_ASSERT_FLOAT_WITHIN(delta, serial_dot, m256_dot);
        TEST_ASSERT_FLOAT_WITHIN(delta, serial_dot_kahan, m256_dot);
#ifdef SUPPORTS_AVX512
        TEST_ASSERT_FLOAT_WITHIN(delta, exact_dot, m512_dot);
        TEST_ASSERT_FLOAT_WITHIN(delta, serial_dot, m512_dot);
        TEST_ASSERT_FLOAT_WITHIN(delta, serial_dot_kahan, m512_dot);
        TEST_ASSERT_FLOAT_WITHIN(delta, m256_dot, m512_dot);
#endif
    }
}

void test_ddots()
{
    int m = 7, n = 100;
    double *x = NULL, *y = NULL, *accumulator = NULL;

    x = calloc(2*m, sizeof(double));

    TEST_ASSERT_NOT_NULL(x);

    if (x != NULL) {
        y = x + m;

        set_darray(x, m, 0.1);
        set_darray(y, m, 1.0);

        double delta;
        double element = x[0];
        double exact_dot = (double)m / 10;
        double serial_dot = serial_ddot(x, y, m);
        double serial_add = serial_dsum(x, m);
        double serial_dot_kahan = serial_ddot_kahan(x, y, m);
        double m256_dot = _mm256_ddot(x, y, m);
#ifdef SUPPORTS_AVX512
        double m512_dot = _mm512_ddot(x, y, m);
#endif
	    free(x);

        delta = 10 * exact_dot * DBL_EPSILON;
        if (delta < 0) delta = -delta;

        printf("element:       %lf\n", element);
        printf("exact:         %lf\n", exact_dot);
        printf("serial:        %lf\n", serial_dot);
        printf("serial sum:    %lf\n", serial_add);
        printf("serial kahan:  %lf\n", serial_dot_kahan);
        printf("m256:          %lf\n", m256_dot);

#ifdef SUPPORTS_AVX512
        printf("m512:          %lf\n", m512_dot);
#endif

        TEST_ASSERT_DOUBLE_WITHIN(delta, exact_dot, serial_dot);
        TEST_ASSERT_DOUBLE_WITHIN(delta, exact_dot, serial_dot_kahan);
        TEST_ASSERT_DOUBLE_WITHIN(delta, exact_dot, m256_dot);
        TEST_ASSERT_DOUBLE_WITHIN(delta, serial_dot, serial_dot_kahan);
        TEST_ASSERT_DOUBLE_WITHIN(delta, serial_dot, m256_dot);
        TEST_ASSERT_DOUBLE_WITHIN(delta, serial_dot_kahan, m256_dot);
#ifdef SUPPORTS_AVX512
        TEST_ASSERT_DOUBLE_WITHIN(delta, exact_dot, m512_dot);
        TEST_ASSERT_DOUBLE_WITHIN(delta, serial_dot, m512_dot);
        TEST_ASSERT_DOUBLE_WITHIN(delta, serial_dot_kahan, m512_dot);
        TEST_ASSERT_DOUBLE_WITHIN(delta, m256_dot, m512_dot);
#endif
    }
}
