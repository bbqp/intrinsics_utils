#!/usr/bin/bash

# Path for the intrinsics library against which we need to link.
libintrinsics_utils_path=${HOME}/repos/intrinsics_utils/lib

if [[ -z "$LD_LIBRARY_PATH" ]];
then
    echo "LD_LIBRARY_PATH is empty, setting LD_LIBARY_PATH to $libintrinsics_utils_path"
    export LD_LIBRARY_PATH=$libintrinsics_utils_path
else
    if [[ :"$LD_LIBRARY_PATH": == *:"$libintrinsics_utils_path":* ]];
    then
        echo "PATH contains $libintrinsics_utils_path, taking no action"
    else
        echo "Appending $libintrinsics_utils_path to LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$libintrinsics_utils_path
    fi
fi

echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"

make clean
make tests
