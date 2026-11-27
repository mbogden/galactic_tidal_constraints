all: compile_notebooks make_spam

# Compile Fortran code in Simulator directory
make_spam:
	cd Simulator;make

# Compile jupyter notebooks into python scripts in misc directories
# pip installs are my short term solutions for a few issues.  Permanent solutions require updating singularity image later.
compile_notebooks:
	pip install --upgrade pip
	pip install --upgrade nbconvert 
	pip install pygad
	pip install scikit-optimize
	pip install astropy
	bash utilities/compile_notebooks.sh Simulator utilities/update_build_env.py
	bash utilities/compile_notebooks.sh IllustrisTNG utilities/update_build_env.py
	bash utilities/compile_notebooks.sh Optimization_Methods utilities/update_build_env.py