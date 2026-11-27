# Makefile to convert Jupyter notebooks to Python scripts

# Define the directory for both notebooks and scripts (current directory)
DIR := .

# Find all notebook files in the current directory
NOTEBOOKS := $(wildcard $(DIR)/*.ipynb)

# Define the corresponding Python script filenames
SCRIPTS := $(patsubst %.ipynb,%.py,$(NOTEBOOKS))

# Default target
all: $(SCRIPTS)

# Rule to convert a notebook to a script
%.py: %.ipynb
	jupyter nbconvert --to script --TagRemovePreprocessor.remove_cell_tags='["dev"]' --TagRemovePreprocessor.enabled=True $< --output $(basename $@)

# Clean up all Python scripts
clean:
	rm -f $(SCRIPTS)

# Help target to display callable targets and their descriptions
help:
	@echo "Available targets:"
	@echo "  all   : Convert all Jupyter notebooks to Python scripts."
	@echo "  clean : Remove all converted Python scripts."
