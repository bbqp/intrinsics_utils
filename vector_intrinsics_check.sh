#!/usr/bin/bash

srcpath=$PWD/src

touch mm512_intrinsics_tmp
touch mm256_intrinsics_tmp
touch mm_intrinsics_tmp


for file in $srcpath/*; do
    bname=$(basename "$file")
    fname="${bname%.*}"

    # Here's the method to the madness:
    # 1. Find all lines with the _mm512 function prefix.
    # 2. Include line with at most one equal sign since we may be making some _mm*store* calls.
    # 3. Remove leading white space.
    # 4. Remove the first portion of variabl declarations.
    # 5. Remove the remaining portion of variable declarations, including in-between white space.
    # 6. Remove the first portion of return statements and return types.
    # 7. Remove opening parentheses on function calls as a prerequisite to awk.
    # 8. Use awk to print the name of the function calls and dump the function names to a temporary file for further processing.
    grep "_mm512" src/intrinsics_utils.c | \
         grep -E "=?" | \
         sed -E 's/^\s*//' | \
         sed -E 's/\s*__m(256|512)d?//' | \
         sed -E 's/^\s*\w+\s\=\s+//' | \
         sed -E 's/^(return|float|double|int|void)//' | \
         sed -E 's/\(/ /' | \
         awk '{print $1}' >> mm512_intrinsics_tmp

    # The same process as above, but for avx/avx2 intrinsics.
    grep "_mm256" src/intrinsics_utils.c | \
         grep -E "=?" | \
         sed -E 's/^\s*//' | \
         sed -E 's/\s*__m(128|256)d?//' | \
         sed -E 's/^\s*\w+\s\=\s+//' | \
         sed -E 's/^(return|float|double|int|void)//' | \
         sed -E 's/\(/ /' | \
         awk '{print $1}' >> mm256_intrinsics_tmp

    # The same process as above, but for mmx/ss*e* intrinsics.
    grep "_mm_" src/intrinsics_utils.c | \
         grep -E "=?" | \
         sed -E 's/^\s*//' | \
         sed -E 's/\s*__m(128|256)d?//' | \
         sed -E 's/^\s*\w+\s\=\s+//' | \
         sed -E 's/^(return|float|double|int|void)//' | \
         sed -E 's/\(/ /' | \
         awk '{print $1}' >> mm_intrinsics_tmp
done

touch mm512_intrinsics
cat mm512_intrinsics_tmp | sort -u > mm512_intrinsics
rm mm512_intrinsics_tmp

touch mm256_intrinsics
cat mm256_intrinsics_tmp | sort -u > mm256_intrinsics
rm mm256_intrinsics_tmp

touch mm_intrinsics
cat mm_intrinsics_tmp | sort -u > mm_intrinsics
rm mm_intrinsics_tmp
