#!/bin/bash

# Script to convert all Jupyter Notebook files to Python scripts in the current directory

# If no arguments are provided, assume the current directory is the target
if [ $# -eq 0 ]; then
    target="."
else
    target="$1"
fi

# Determine if target is a single notebook file or a directory
if [ -f "$target" ] && [[ "$target" == *.ipynb ]]; then
    notebooks=("$target")  # Single file
elif [ -d "$target" ]; then
    notebooks=("$target"/*.ipynb)  # All notebooks in the directory
else
    echo "Error: $target is not a valid notebook file or directory."
    exit 1
fi

# Check if any .ipynb files were found
if [ ! -e "${notebooks[0]}" ]; then
    echo "No .ipynb files found in $target."
    exit 0
fi

echo ""

# Loop through the notebooks and process them
for notebook in "${notebooks[@]}"; do

    # create name of python script
    script="${notebook%.ipynb}.py"
    echo "Converting $notebook to $script"

    # Use jupyter nbconvert to convert them to Python scripts
    jupyter nbconvert --to=python --TagRemovePreprocessor.remove_cell_tags dev --TagRemovePreprocessor.enabled=True "$notebook"

    echo ""
    
done