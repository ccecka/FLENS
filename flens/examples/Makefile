#########################
## Library Directories ##
########################

# Executables
EXEC = $(basename $(wildcard *.cc))

# Remove select executables
EXEC := $(filter-out design-flaws-mtl4, $(EXEC))
EXEC := $(filter-out geccs-superlu gecrs-superlu, $(EXEC))
EXEC := $(filter-out mpfr-real-lapack-geev mpfr-real-lapack-getrf, $(EXEC))
EXEC := $(filter-out qd-lapack-geev qd-lapack-getrf, $(EXEC))

EXEC := $(filter-out tut01-page03-example04 tut02-page03-example tut02-page04-example1 tut02-page04-example2 tut05-page02-example2 tut05-page02-example4 tut08-page01-example2 tut08-page01-example5, $(EXEC))

# External libraries
FLENS_DIR = ../../

####################
## Makefile Setup ##
####################

# Get the host-name if empty
ifeq ($(host-name),)
	host-name := $(shell hostname)
endif
# Get the kernel-name if empty
ifeq ($(kernel-name),)
	kernel-name := $(shell uname -s)
endif
# Get the arch-name if empty
ifeq ($(arch-name),)
	arch-name := $(shell uname -p)
endif

# Define the C++ compiler to use
#CXX := $(shell which g++) -std=c++11

# Dependency directory and flags
DEPSDIR := $(shell mkdir -p .deps; echo .deps)
# MD: Dependency as side-effect of compilation
# MF: File for output
# MP: Include phony targets
DEPSFILE = $(DEPSDIR)/$(notdir $*.d)
DEPSFLAGS = -MD -MF $(DEPSFILE) #-MP

# Define any directories containing header files
#   To include directories use -Ipath/to/files
INCLUDES += -I. -I$(FLENS_DIR)

# Define cxx compile flags
CXXFLAGS = $(CFLAGS) -std=c++11 -O3 -Wall -Wextra

# Define any directories containing libraries
#   To include directories use -Lpath/to/files
LDFLAGS +=

# Define any libraries to link into executable
#   To link in libraries (libXXX.so or libXXX.a) use -lXXX
LDLIBS  +=

######################
## Makefile Options ##
######################


####################
## Makefile Rules ##
####################

#   $^: the name of the prereqs of the rule
#   $<: the name of the first prereq of the rule
#   $@: the name of the target of the rule

# 'make' - default rule
all: $(EXEC)

# Default rule for creating an exec of $(EXEC) from a .o file
$(EXEC): % : %.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Default rule for creating a .o file from a .cc file
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPSFLAGS) -c -o $@ $<

# 'make clean' - deletes all .o and temp files, exec, and dependency file
clean:
	-$(RM) *.o
	-$(RM) $(EXEC)
	$(RM) -r $(DEPSDIR)

# Define rules that do not actually generate the corresponding file
.PHONY: clean all

# Include the dependency files
-include $(wildcard $(DEPSDIR)/*.d)
