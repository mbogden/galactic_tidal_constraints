# Makefile to compile Fortran wrapper code into a Python module using f2py

# Variables
WRAPPER_CODE = spam_runs.f90
MODULE_NAME = spam_runs
F2PY = f2py
SPAM_OBJ_DIR = ../Simulator/fortran/obj
MOD_DIR = ../Simulator/fortran/build
SPAM_OBJECTS = $(SPAM_OBJ_DIR)/parameters_module.o $(SPAM_OBJ_DIR)/io_module.o \
				$(SPAM_OBJ_DIR)/df_module.o $(SPAM_OBJ_DIR)/setup_module.o \
				$(SPAM_OBJ_DIR)/integrator.o $(SPAM_OBJ_DIR)/init_module.o

# Default target
all: $(MODULE_NAME)

$(MODULE_NAME): $(WRAPPER_CODE) $(SPAM_OBJECTS)
	$(F2PY) -c -m $(MODULE_NAME) $(WRAPPER_CODE) $(SPAM_OBJECTS) --fcompiler=gfortran \
        -I$(MOD_DIR) -lgfortran

# Clean the compiled modules
clean:
	rm -f $(MODULE_NAME)*.so

.PHONY: all clean
