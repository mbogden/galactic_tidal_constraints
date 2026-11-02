# Running SPAM

How to compile and run SPAM, the galaxy collision simulator this project depends on. SPAM's code
is in [JSPAM](https://github.com/mbogden/JSPAM), a fork of
[John Wallin's original](https://github.com/jfwallin/JSPAM). We use its Fortran version.

This runs the unmodified Fortran from the command line. It can't be installed as a Python package
yet; this page will change when it can.

## What you need
- [Docker](https://www.docker.com/), which provides the compiler through this project's base image.

## Quick check
To confirm the original code works on your machine, clone JSPAM (step 1) and run this from this
repo's root, using bash (Git Bash on Windows):
```bash
bash examples/spam/check_original_jspam.sh
```
It does steps 2–5 below for you, on Wallin's original commit, and exits with an error if the result
is wrong. Follow the steps below to do it by hand.

## 1. Get the code
Clone JSPAM next to this repo:
```
galStuff/
├── galactic_tidal_constraints/
└── JSPAM/
```
```bash
git clone https://github.com/mbogden/JSPAM.git
```

## 2. Start the container
From the folder that contains `JSPAM/` (this command works in bash and PowerShell):
```bash
docker run --rm -it -v "${PWD}/JSPAM:/src:ro" ogdenm12/gtc-base-image:v1 bash
```
`JSPAM/` is mounted read-only, so the build stays out of your clone. Anything in `/tmp/spam` is
lost when you exit the container.

## 3. Compile
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

## 4. Run
```bash
./basic_run > run.log
```
With no arguments, this runs a built-in test collision of 1000 + 500 particles, using the
point-mass gravity model. It should take about a second.

## 5. Check the result
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
