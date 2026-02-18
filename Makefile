# Possible combinations of compiler flags.
# Serial
# Using Indices: -DUSE_INDICES
# Using Intrinsics: -DUSE_INTRINSICS
# Using Intrinsics and Indices: -DUSE_INTRINSICS -DUSE_INDICES
# Using Intrinsics with Contiguous Loop for Storage: -DUSE_INTRINSICS -DCONTIGUOUS_LOOP
# Using Indices and Intrinsics with Contiguous Loop for Storage: -DUSE_INTRINSICS -DUSE_INDICES -DCONTIGUOUS_LOOP

# The clear winner -> -O2 -march=native -DUSE_INTRINSICS -DUSE_INDICES -DCONTIGUOUS_LOOP

# Name of the shared object file.
SONAME=libintrinsics_utils
SOMMP=0
LIBMMP=0.0.0

ifeq ($(OS),Windows_NT)
	SOEXT=dll
	SONAMEEXT=dll
	LIBNAMEEXT=dll
else
	SOEXT=so
	SONAMEEXT=${SOEXT}.${SOMMP}
	LIBNAMEEXT=${SOEXT}.${LIBMMP}
endif

cc=gcc
ccflags=-fPIC -march=native -I$(include_dir) -DCONTIGUOUS_LOOP
ldflags=-shared -Wl,-soname,${SONAME}.${SONAMEEXT}

src_dir=$(PWD)/src/
include_dir=$(PWD)/include/
object_dir=$(PWD)/obj/
bin_dir=$(PWD)/bin/
lib_dir=$(PWD)/lib/

src_files=$(wildcard $(src_dir)/*.c)
obj_files=$(patsubst $(src_dir)/%.c, $(object_dir)/%.o, $(src_files))

intrinsics_utils: $(object_dir) $(bin_dir) $(lib_dir) $(obj_files)
	$(cc) $(obj_files) -o $(lib_dir)/${SONAME}.${LIBNAMEEXT} $(ldflags)
	-ln -s $(lib_dir)/${SONAME}.${LIBNAMEEXT} $(lib_dir)/${SONAME}.${SONAMEEXT}
	-ln -s $(lib_dir)/${SONAME}.${SONAMEEXT} $(lib_dir)/${SONAME}.${SOEXT}

$(object_dir)/intrinsics_utils.o: $(src_dir)/intrinsics_utils.c $(include_dir)/intrinsics_utils.h $(include_dir)/mask_utils.h $(include_dir)/constants.h $(include_dir)/cpu_flags.h
	$(cc) -c $< $(ccflags) -o $@ 

$(object_dir)/mask_utils.o: $(src_dir)/mask_utils.c $(include_dir)/mask_utils.h $(include_dir)/constants.h $(include_dir)/cpu_flags.h
	$(cc) -c $< $(ccflags) -o $@ 

$(object_dir):
	mkdir -p $(object_dir)

$(bin_dir):
	mkdir -p $(bin_dir)

$(lib_dir):
	mkdir -p $(lib_dir)

clean:
	rm -rf $(object_dir) $(bin_dir) $(lib_dir)

.PHONY: clean, setup
