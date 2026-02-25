#include "unity.h"
#include "mask_utils.h"
#include "intrinsics_utils.h"
#include <stdlib.h>
#include <float.h>

#define FLT_DELTA (2e2 * FLT_EPSILON)
#define DBL_DELTA (2e2 * DBL_EPSILON)

// Global variables for performing dot products.
float *xf = NULL, *yf = NULL;
double *xd = NULL, *yd = NULL;
int *xindices = NULL;
int *yindices = NULL;

// Array dimensions.
int m = 1000;
int n = 1000;

// Random seed for srand call. 
unsigned random_seed = 0;

// Forward declarations needed to use Unity.
void setUp(void);
void tearDown(void);

// Forward declarations for initializing arrays and computing dot products.
void random_farray(float *, int, float, float);
void set_farray(float *, int, float);
void seq_farray(float *, int, float, float);
void copy_farray(float *, const float *, int);
float serial_fdot(const float *, const float *, int);
float serial_fsum(const float *, int);
float serial_fdot_kahan(const float *, const float *, int);

// Forward declarations for double precision functions.
void random_darray(double *, int, double, double);
void set_darray(double *, int, double);
void seq_darray(double *, int, double, double);
void copy_darray(double *, const double *, int);
double serial_ddot(const double *, const double *, int);
double serial_dsum(const double *, int);
double serial_ddot_kahan(const double *, const double *, int);

// Forward declarations fo setting indices.
void random_index_array(int *, int);
void seq_index_array(int *, int, int, int);

// Forward declarations for tests.
void test_m256_fdot(void);
void test_m256_fdot_indexed(void);
void test_m256_ddot(void);
void test_m256_ddot_indexed(void);

#ifdef SUPPORTS_AVX512
void test_m512_fdot(void);
void test_m512_ddot(void);
#endif

int main(int argc, char *argv[])
{
    if (argc > 1) {
        m = strtol(argv[1], NULL, 10);
    }

    if (argc > 2) {
        n = strtol(argv[2], NULL, 10);
    }

    if (argc > 3) {
        random_seed = strtoul(argv[3], NULL, 10);
    }

    UNITY_BEGIN();

    RUN_TEST(test_m256_fdot);
    RUN_TEST(test_m256_fdot_indexed);
    RUN_TEST(test_m256_ddot);
    RUN_TEST(test_m256_ddot_indexed);

#ifdef SUPPORTS_AVX512
    RUN_TEST(test_m512_fdot);
    RUN_TEST(test_m512_ddot);
#endif

    return UNITY_END();
}

void setUp(void)
{
    srand(random_seed);

    xf = calloc(2*m, sizeof(float));
    xd = calloc(2*m, sizeof(double));
    xindices = calloc(2*m, sizeof(int));
    
    if (xf != NULL && xd != NULL && xindices != NULL) {
        yf = xf + m;
        yd = xd + m;
        yindices = xindices + m;
    } else {
        tearDown();
    }
}

void tearDown(void)
{
    free(xf);
    free(xd);
    free(xindices);

    xf = yf = NULL;
    xd = yd = NULL;
    xindices = yindices = NULL;
}

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

void seq_farray(float *x, int len, float start, float step)
{
    for (int i = 0; i < len; i++) {
        x[i] = start + i * step;
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

void seq_darray(double *x, int len, double start, double step)
{
    for (int i = 0; i < len; i++) {
        x[i] = start + i * step;
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

// Functions for setting array indices.
void random_index_array(int *indices, int len)
{
    int j;
    int temp;

    // Use the first loop to populate the array.
    seq_index_array(indices, len, 0, 1);

    // Use the modernized implementation of the Fischer-Yates algorithm.
    for (int i = len - 1; i >= 0; i--) {
        j = rand() % (i + 1);

        if (i != j) {
            temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        } 
    }
}

void seq_index_array(int *indices, int len, int start, int step)
{
    for (int i = 0; i < len; i++) {
        indices[i] = start + i * step;
    }
}

//----------------------------------------------------------------------------
// Tests for intrinsics.
//----------------------------------------------------------------------------

void test_m256_fdot(void)
{
    set_farray(xf, m, 0.1f);
    set_farray(yf, m, 1.0f);

    float element = xf[0];
    float exact_dot = (float)m * element;
    float serial_dot = serial_fdot(xf, yf, m);
    float serial_add = serial_fsum(xf, m);
    float serial_dot_kahan = serial_fdot_kahan(xf, yf, m);
    float m256_dot = _mm256_fdot(xf, yf, m);
    float m256_rel_err = (m256_dot - exact_dot) / exact_dot;

    TEST_PRINTF("element:                          %f", element);
    TEST_PRINTF("exact:                            %f", exact_dot);
    TEST_PRINTF("serial:                           %f", serial_dot);
    TEST_PRINTF("serial sum:                       %f", serial_add);
    TEST_PRINTF("serial kahan:                     %f", serial_dot_kahan);
    TEST_PRINTF("m256:                             %f", m256_dot);
    TEST_PRINTF("m256 relative error:              %f", m256_rel_err);

    if (m256_rel_err < 0) m256_rel_err = -m256_rel_err;
    TEST_ASSERT_LESS_THAN_FLOAT(FLT_DELTA, m256_rel_err);
}

void test_m256_fdot_indexed(void)
{
    int seq_len = 20;

    seq_farray(xf, seq_len, 1.0f, 1.0f);
    set_farray(yf, seq_len, 1.0f);
    random_index_array(xindices, seq_len);

    float exact_dot = (float)((seq_len * (seq_len + 1)) / 2);
    float serial_dot = serial_fdot(xf, yf, seq_len);
    float serial_add = serial_fsum(xf, seq_len);
    float serial_dot_kahan = serial_fdot_kahan(xf, yf, seq_len);
    float m256_dot = _mm256_fdot(xf, yf, seq_len);
    float m256_dot_idx = _mm256_fdot_indexed(xf, xindices, yf, seq_len);
    float m256_rel_err = (m256_dot_idx - exact_dot) / exact_dot;

    printf("Randomized Indices: ");
    for (int i = 0; i < seq_len; i++) {
        printf("%d ", xindices[i]);
    }
    printf("\n");

    TEST_PRINTF("exact:                            %f", exact_dot);
    TEST_PRINTF("serial:                           %f", serial_dot);
    TEST_PRINTF("serial sum:                       %f", serial_add);
    TEST_PRINTF("serial kahan:                     %f", serial_dot_kahan);
    TEST_PRINTF("m256:                             %f", m256_dot);
    TEST_PRINTF("m256 random indices:              %f", m256_dot_idx);
    TEST_PRINTF("m256 relative error:              %f", m256_rel_err);

    if (m256_rel_err < 0) m256_rel_err = -m256_rel_err;
    TEST_ASSERT_LESS_THAN_FLOAT(FLT_DELTA, m256_rel_err);   
}

void test_m256_ddot(void)
{
    set_darray(xd, m, 0.1);
    set_darray(yd, m, 1.0);

    double element = xd[0];
    double exact_dot = (double)m * element;
    double serial_dot = serial_ddot(xd, yd, m);
    double serial_add = serial_dsum(xd, m);
    double serial_dot_kahan = serial_ddot_kahan(xd, yd, m);
    double m256_dot = _mm256_ddot(xd, yd, m);
    double m256_rel_err = (m256_dot - exact_dot) / exact_dot;

    TEST_PRINTF("element:                          %lf", element);
    TEST_PRINTF("exact:                            %lf", exact_dot);
    TEST_PRINTF("serial:                           %lf", serial_dot);
    TEST_PRINTF("serial sum:                       %lf", serial_add);
    TEST_PRINTF("serial kahan:                     %lf", serial_dot_kahan);
    TEST_PRINTF("m256:                             %lf", m256_dot);
    TEST_PRINTF("m256 relative error:              %lf", m256_rel_err);

    if (m256_rel_err < 0) m256_rel_err = -m256_rel_err;
    TEST_ASSERT_LESS_THAN_DOUBLE(DBL_DELTA, m256_rel_err);
}

void test_m256_ddot_indexed(void)
{
    int seq_len = 20;

    seq_darray(xd, seq_len, 1.0, 1.0);
    set_darray(yd, seq_len, 1.0);
    random_index_array(xindices, seq_len);

    double exact_dot = (double)((seq_len * (seq_len + 1)) / 2);
    double serial_dot = serial_ddot(xd, yd, seq_len);
    double serial_add = serial_dsum(xd, seq_len);
    double serial_dot_kahan = serial_ddot_kahan(xd, yd, seq_len);
    double m256_dot = _mm256_ddot(xd, yd, seq_len);
    double m256_dot_idx = _mm256_ddot_indexed(xd, xindices, yd, seq_len);
    double m256_rel_err = (m256_dot_idx - exact_dot) / exact_dot;

    printf("Randomized Indices: ");
    for (int i = 0; i < seq_len; i++) {
        printf("%d ", xindices[i]);
    }
    printf("\n");

    TEST_PRINTF("exact:                            %lf", exact_dot);
    TEST_PRINTF("serial:                           %lf", serial_dot);
    TEST_PRINTF("serial sum:                       %lf", serial_add);
    TEST_PRINTF("serial kahan:                     %lf", serial_dot_kahan);
    TEST_PRINTF("m256:                             %lf", m256_dot);
    TEST_PRINTF("m256 random indices:              %lf", m256_dot_idx);
    TEST_PRINTF("m256 relative error:              %lf", m256_rel_err);

    if (m256_rel_err < 0) m256_rel_err = -m256_rel_err;
    TEST_ASSERT_LESS_THAN_FLOAT(DBL_DELTA, m256_rel_err);   
}

#ifdef SUPPORTS_AVX512
void test_m512_fdot(void)
{
    set_farray(xf, m, 0.1f);
    set_farray(yf, m, 1.0f);

    float element = xf[0];
    float exact_dot = (float)m * element;
    float serial_dot = serial_fdot(xf, yf, m);
    float serial_add = serial_fsum(xf, m);
    float serial_dot_kahan = serial_fdot_kahan(xf, yf, m);
    float m512_dot = _mm512_fdot(xf, yf, m);
    float m512_rel_err = (m512_dot - exact_dot) / exact_dot;

    TEST_PRINTF("element:                          %f", element);
    TEST_PRINTF("exact:                            %f", exact_dot);
    TEST_PRINTF("serial:                           %f", serial_dot);
    TEST_PRINTF("serial sum:                       %f", serial_add);
    TEST_PRINTF("serial kahan:                     %f", serial_dot_kahan);
    TEST_PRINTF("m512:                             %f", m512_dot);
    TEST_PRINTF("m512 relative error:              %f", m512_rel_err);

    if (m512_rel_err < 0) m512_rel_err = -m512_rel_err;
    TEST_ASSERT_LESS_THAN_FLOAT(FLT_DELTA, m512_rel_err);
}

void test_m512_ddot(void)
{
        set_darray(xd, m, 0.1);
        set_darray(yd, m, 1.0);

        double element = xd[0];
        double exact_dot = (double)m * element;
        double serial_dot = serial_ddot(xd, yd, m);
        double serial_add = serial_dsum(xd, m);
        double serial_dot_kahan = serial_ddot_kahan(xd, yd, m);
        double m512_dot = _mm512_ddot(xd, yd, m);
        double m512_rel_err = (m512_dot - exact_dot) / exact_dot;

        TEST_PRINTF("element:                          %lf", element);
        TEST_PRINTF("exact:                            %lf", exact_dot);
        TEST_PRINTF("serial:                           %lf", serial_dot);
        TEST_PRINTF("serial sum:                       %lf", serial_add);
        TEST_PRINTF("serial kahan:                     %lf", serial_dot_kahan);
        TEST_PRINTF("m512:                             %lf", m512_dot);
        TEST_PRINTF("m512 relative error:              %lf", m512_rel_err);

        if (m512_rel_err < 0) m512_rel_err = -m512_rel_err;
        TEST_ASSERT_LESS_THAN_DOUBLE(DBL_DELTA, m512_rel_err);
}
#endif
