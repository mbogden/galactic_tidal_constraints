#!/bin/bash

# Script to convert all Jupyter Notebook files to Python scripts in the current directory

# If no arguments are provided, assume the current directory is the target
if [ $# -eq 0 ]; then
    target="."
else
    target="$1"
fi



# If not in curren dir, check utilities dir
if [ ! -f "$update_location" ]; then
    update_location="utilities/update_build_env.py"
fi

# if not in utilities dir, check parent/utilities dir
if [ ! -f "$update_location" ]; then
    update_location="../utilities/update_build_env.py"
fi

# If not found, check for a 2nd argument
if [ ! -f "$update_location" ]; then
    if [ $# -eq 2 ]; then
        update_location="$2"
    fi
fi

# exit if not found in any of the directories
if [ ! -f "$update_location" ]; then
    echo "update_build_env.py not found in current directory or utilities directory."
    exit 1
fi
    

# Find all .ipynb files in the target directory and loop through them
for notebook in "$target"/*.ipynb; do
    # Use jupyter nbconvert to convert them to Python scripts
    jupyter nbconvert --to script "$notebook"
    
    # Extract the Python script filename from the notebook filename
    script="${notebook%.ipynb}.py"

    # Update the Python script to include the build environment
    python "$update_location" "$script"
    
    echo "Converted $notebook to Python script."
done