# Running SPAM

How to compile and run SPAM, the galaxy collision simulator this project depends on. SPAM's code
is in [JSPAM](https://github.com/mbogden/JSPAM), a fork of
[John Wallin's original](https://github.com/jfwallin/JSPAM). We use its Fortran version.

There are two ways to run it: from Python, by installing the `jspam` package, or as the plain Fortran
program from the command line. This project will use the Python package.

## Run it from Python
Install it directly from GitHub. You need Python 3.12 or newer, git, `gfortran` and a C compiler:
```bash
pip install git+https://github.com/mbogden/JSPAM.git@simr-branch
```
The fork's [README](https://github.com/mbogden/JSPAM/tree/simr-branch#use) has a short example.

This project's base image has everything needed, so you can use it instead:
```bash
docker run --rm -it ogdenm12/gtc-base-image:v2 bash
uv pip install git+https://github.com/mbogden/JSPAM.git@simr-branch
```
The second command runs inside the container.

## Check the original code
To confirm the original code works on your machine, install git and
[Docker](https://www.docker.com/), clone JSPAM (step 1) and run this from this repo's root, using
bash (Git Bash on Windows):
```bash
bash examples/spam/check_original_jspam.sh
```
It does steps 2–5 below for you, on Wallin's original commit, and exits with an error if the result
is wrong.

## Run the Fortran program by hand
You need [Docker](https://www.docker.com/), which provides the compiler through the base image.

### 1. Get the code
Clone JSPAM next to this repo:
```
galStuff/
├── galactic_tidal_constraints/
└── JSPAM/
```
```bash
git clone https://github.com/mbogden/JSPAM.git
```

### 2. Start the container
From the folder that contains `JSPAM/` (this command works in bash and PowerShell):
```bash
docker run --rm -it -v "${PWD}/JSPAM:/src:ro" ogdenm12/gtc-base-image:v2 bash
```
`JSPAM/` is mounted read-only, so the build stays out of your clone. Anything in `/tmp/spam` is
lost when you exit the container.

### 3. Compile
Run these inside the container:
```bash
mkdir /tmp/spam && cp /src/fortran/*.f90 /tmp/spam && cd /tmp/spam
for m in parameters_module io_module df_module setup_module integrator init_module basic_run; do
    gfortran -O3 -std=legacy -c $m.f90
done
gfortran -O3 -o basic_run *.o
```
- The modules must be compiled in this order, because each one needs the ones before it.
- `-std=legacy` silences warnings about older Fortran syntax.
- `-O3` matters for speed. Without an optimization flag, `gfortran` builds unoptimized code.

### 4. Run
```bash
./basic_run > run.log
```
With no arguments, this runs a built-in test collision of 1000 + 500 particles, using the
point-mass gravity model. It should take about a second.

### 5. Check the result
The final particle positions and velocities are in `a.101`, one row per particle:
```bash
wc -l a.101
tail -1 a.101
```
Expected output (the numbers may vary slightly with a different compiler or flags):
```
1501 a.101
  0.89999667E+00 -0.29296680E-02 -0.10663120E-02 -0.33333221E-02 -0.14648304E+01 -0.53315466E+00
```
The last row is the second galaxy's center (x, y, z, vx, vy, vz), which should match on every run.
The rows above it are disk particles, which start from unseeded random positions and differ
between runs.
