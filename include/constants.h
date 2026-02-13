#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>

//----------------------------------------------------------------------------
// Macros for imm8 constants for permuting/shuffling.
//----------------------------------------------------------------------------

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
// Macros for deciding between column- and row-major ordering.
//----------------------------------------------------------------------------

#define COLUMN_MAJOR_ORDER 'c'
#define ROW_MAJOR_ORDER 'r'

//----------------------------------------------------------------------------
// Macros for regioster sizes and masking.
//----------------------------------------------------------------------------

#define DOUBLE_PER_M128_REG 2
#define INT64_PER_M128_REG DOUBLE_PER_M128_REG
#define DOUBLE_PER_M256_REG 4
#define INT64_PER_M256_REG DOUBLE_PER_M256_REG
#define DOUBLE_PER_M512_REG 8
#define INT64_PER_M512_REG DOUBLE_PER_M512_REG

#define FLOAT_PER_M128_REG 4
#define INT32_PER_M128_REG FLOAT_PER_M128_REG
#define FLOAT_PER_M256_REG 8
#define INT32_PER_M256_REG FLOAT_PER_M256_REG
#define FLOAT_PER_M512_REG 16
#define INT32_PER_M512_REG FLOAT_PER_M512_REG

#define INT32_ZERO ((int32_t)0)
#define INT32_ALLBITS ((int32_t)0xFFFFFFFF)
#define INT32_LOWBIT  ((int32_t)0x00000001)
#define INT32_HIGHBIT ((int32_t)0x80000000)

#define INT64_ZERO    ((int64_t)0)
#define INT64_LOWBIT  ((int64_t)0x0000000000000001)
#define INT64_HIGHBIT ((int64_t)0x8000000000000000)
#define INT64_ALLBITS ((int64_t)0xFFFFFFFFFFFFFFFF)

#endif
