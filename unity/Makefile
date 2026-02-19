# Makefile for testing intrinsics utilities using the Unity unit testing tool
# for C.

# Begin by detecting the operating system.
CLEANUP=rm -f
MKDIR=mkdir -p
TARGET_EXTENSION=out

# Point to the Unity source, as well as the source, include, and shared
# library paths for this repo.
ifeq ($(OS),Windows_NT)
	BASEPATH=/c/Users/$(USER)/
else
	BASEPATH=$(HOME)/repos/
endif

PATHU=$(BASEPATH)/Unity/src/
PATHS=$(BASEPATH)/intrinsics_utils/src/
PATHI=$(BASEPATH)/intrinsics_utils/include/
PATHL=$(BASEPATH)/intrinsics_utils/lib/

# Paths needed for test source, build, dependencies, object files, and test
# results.
PATHT=./
PATHB=./build/
PATHD=./build/depends/
PATHO=./build/objs/
PATHR=./build/results/

BUILD_PATHS = $(PATHB) $(PATHD) $(PATHO) $(PATHR)

# Grab all of the testing source files.
SRCT=$(wildcard $(PATHT)*.c)

# Compilation flags needed. Modify these to fit with this repo.
CC=gcc
COMPILE=$(CC) -c -march=native
LINK=$(CC)
LDFLAGS=-L$(HOME)/repos/intrinsics_utils/lib/ -lintrinsics_utils
DEPEND=$(CC) -MM -MG -MF
CFLAGS=-I$(PATHI) -I$(PATHU) -DTEST

RESULTS=$(patsubst $(PATHT)Test%.c,$(PATHR)Test%.txt,$(SRCT))

tests: $(BUILD_PATHS) $(RESULTS)
	@echo -e "-----------------------\nIGNORES:\n-----------------------"
	@echo `grep -s IGNORE $(PATHR)*.txt`
	@echo -e "-----------------------\nFAILURES:\n-----------------------"
	@echo `grep -s FAIL $(PATHR)*.txt`
	@echo -e "-----------------------\nPASSED:\n-----------------------"
	@echo `grep -s PASS $(PATHR)*.txt`
	@echo -e "\nDONE"

$(PATHR)%.txt: $(PATHB)%.$(TARGET_EXTENSION)
	-./$< > ./$@ 2>&1

$(PATHB)Test%.$(TARGET_EXTENSION): $(PATHO)Test%.o $(PATHO)%.o $(PATHO)unity.o
	@echo "Linking"
	$(LINK) $^ -o $@ $(LDFLAGS)

$(PATHO)%.o:: $(PATHT)%.c
	@echo "Compiling test"
	$(COMPILE) $(CFLAGS) $< -o $@

$(PATHO)%.o:: $(PATHS)%.c
	@echo "Compiling source"
	$(COMPILE) $(CFLAGS) $< -o $@

$(PATHO)%.o:: $(PATHU)%.c $(PATHU)%.h
	@echo "Compiling Unity source"
	$(COMPILE) $(CFLAGS) $< -o $@

$(PATHD)%.d:: $(PATHT)%.c
	@echo "Compiling Test dependencies"
	$(DEPEND) $@ $<

# Create the testing subdirectories.
$(PATHB):
	$(MKDIR) $(PATHB)

$(PATHD):
	$(MKDIR) $(PATHD)

$(PATHO):
	$(MKDIR) $(PATHO)

$(PATHR):
	$(MKDIR) $(PATHR)

clean:
	$(CLEANUP) $(PATHO)*.o
	$(CLEANUP) $(PATHB)*.$(TARGET_EXTENSION)
	$(CLEANUP) $(PATHR)*.txt

.PRECIOUS: $(PATHB)Test%.$(TARGET_EXTENSION)
.PRECIOUS: $(PATHD)%.d
.PRECIOUS: $(PATHO)%.o
.PRECIOUS: $(PATHR)%.txt

.PHONY: clean test
