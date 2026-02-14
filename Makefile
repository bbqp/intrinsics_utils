# Possible combinations of compiler flags.
# Serial
# Using Indices: -DUSE_INDICES
# Using Intrinsics: -DUSE_INTRINSICS
# Using Intrinsics and Indices: -DUSE_INTRINSICS -DUSE_INDICES
# Using Intrinsics with Contiguous Loop for Storage: -DUSE_INTRINSICS -DCONTIGUOUS_LOOP
# Using Indices and Intrinsics with Contiguous Loop for Storage: -DUSE_INTRINSICS -DUSE_INDICES -DCONTIGUOUS_LOOP

# The clear winner -> -O2 -march=native -DUSE_INTRINSICS -DUSE_INDICES -DCONTIGUOUS_LOOP

src_dir=src
include_dir=include
object_dir=obj
bin_dir=bin
lib_dir=lib

library_obj:=$(library_obj:%=$(object_dir)/%)

so_ccflags=-O2 -march=native -I$(include_dir) -DUSE_INDICES -DCONTIGUOUS_LOOP -fPIC
so_ldflags=-shared

# Debug flags
#ccflags=-O2 -march=native -g -pg -no-pie
#ldflags=-pg -no-pie

intrinsic_utils: $(library_obj)
	$(cc) $^ -o $(lib_dir)/libintrinsics_utils.so $(so_ldflags)

$(object_dir)/intrinsics_utils.o: $(src_dir)/intrinsics_utils.c $(include_dir)/intrinsics_utils.h $(include_dir)/mask_utils.h $(include_dir)/constants.h $(include_dir)/cpu_flags.h
	$(cc) -c $< $(ccflags) -o $@ 

$(object_dir)/mask_utils.o: $(src_dir)/mask_utils.c $(include_dir)/mask_utils.h $(include_dir)/constants.h $(include_dir)/cpu_flags.h
	$(cc) -c $< $(ccflags) -o $@ 

clean:
	rm -rf $(object_dir) $(bin_dir) $(lib_dir)

setup:
	mkdir -p $(object_dir)
	mkdir -p $(bin_dir)
	mkdir -p $(lib_dir)

.PHONY: clean, run_validate, run_benchmark
