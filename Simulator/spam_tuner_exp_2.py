#!/usr/bin/env python
# coding: utf-8

#  # Bridging GA runs to HPC jobs
# 
#  ## Getting Base Cmd Working
#  Okay!  So I go overwhelmed trying to remember how to get the HPC up and running when i have a CMD line I know works.  So here is a simple notebook to help me remember how to do that!
# 
#  The golden cmdline I know works is:
# 
#  ```
#  python3 /home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/compare_tng_to_spam.py --function exp_1 --target_id 62000000190717 --input balance_base --input2 test --output /home/mbo2d/galStuff/data/spam_tuner_exp_1
#  ```
# 
# Go ahead and run this in a cmdline after calling
# ```
# ./scripts/sing-wr.sh
# ```
# 
# 
# You'll probably get a File exists and exists early, add --overwrite to the cmdline to fix that.
# 
#     ``` 
#     python3 /home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/compare_tng_to_spam.py --function exp_1 --target_id 62000000190717 --input balance_base --input2 test --output /home/mbo2d/galStuff/data/spam_tuner_exp_1 --overwrite
#     ```
# It'll auto detect it's running on a machine with X cores and auto allocates those cores for the run. And it should output something like this. 
# 
# ```
# [2026-01-09 14:00:46,748 - INFO - compare_tng_to_spam.func_experiment_1]: SPAM Tuner Setup Complete
# [2026-01-09 14:00:47,780 - INFO - parallel_queue_manager._run_local_processes]: Local Manager: Progress: [=                   ] 2/30
# [2026-01-09 14:00:48,274 - INFO - parallel_queue_manager._run_local_processes]: Local Manager: Progress: [====================] 30/30 - Queue Complete
# [2026-01-09 14:00:49,063 - INFO - parallel_queue_manager._run_local_processes]: Local Manager: Progress: [====================] 25/25 - Queue Complete
# [2026-01-09 14:00:49,065 - INFO - compare_tng_to_spam.func_experiment_1]: GA Run Complete
# ```
# 

# ## Singularity Wrapper
# 
# Next, we need this to run in a singularity wrapper.   We were already running in a singularity wrapper above, but now we need to make a script to launch it.  Here is what the script at this location should looking like
# 
# loc: /home/mbo2d/galStuff/scripts/sing-wr.bash
# ```
# #!/bin/bash
# 
# cmd="./$1"
# echo "sing-wr.sh: $cmd"
# 
# # Load singularity
# module load singularity
# 
# # singularity exec --nv /home/mbo2d/images/simr-image-0.sif $cmd
# singularity exec --nv /home/mbo2d/images/babbage-beta-2.sif $cmd
# 
# ```
# 
# Note, All it does is take the argument it's given, loads singularity in the HPC environment, and runs it within the singularity exec command.  Try it now, outside a singularity shell.   Run
# 
# ```
# ./scripts/sing-wr.bash "python3 /home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/compare_tng_to_spam.py --function exp_1 --target_id 62000000190717 --input balance_base --input2 test --output /home/mbo2d/galStuff/data/spam_tuner_exp_1 --overwrite"
# ```
# 
# It'll output something very similar to above, but now it's running in singularity!
# 
# ```
# [2026-01-09 14:39:21,589 - INFO - compare_tng_to_spam.func_experiment_1]: Results saved to /home/mbo2d/galStuff/data/spam_tuner_exp_1/Exp0_target=62000000190717__score=balance_base__ga=test_results.png
# INFO:    Terminating squashfuse_ll after timeout
# INFO:    Timeouts can be caused by a running background process
# ```

# ## Slurm Command.
# 
# Next we need to create a SLURM script to launch this on the HPC.  See below script that will create the file for you!  We'll remove the overwrite flag since we want to keep results from multiple runs.
# 
# Let's save that to disk for you!

# In[1]:


slurm_batch_wrapper_txt = """#!/bin/bash
#SBATCH -J spam_exp_1-2
#SBATCH -p research-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -o /home/mbo2d/galStuff/slurm_batch_logs/spam_exp_1-2/%x_%j.out
#SBATCH -e /home/mbo2d/galStuff/slurm_batch_logs/spam_exp_1-2/%x_%j.err

set -euo pipefail

tid="${1:?target_id required}"
score="${2:?score key required}"
ga="${3:?ga key required}"
out="${4:?output dir required}"

module load singularity
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CMD="python3 /home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/compare_tng_to_spam.py \
  --function exp_1 --target_id ${tid} --input ${score} --input2 ${ga} \
  --output ${out}"

echo "Running: $CMD"
singularity exec --nv /home/mbo2d/images/babbage-beta-2.sif bash -lc "$CMD"
"""

batch_dir = '/home/mbo2d/galStuff/slurm_batch_scripts/'

# create dir if it doesn't exist
os.makedirs( batch_dir, exist_ok=True )


# Let's save the batch script
file_name = "spam_ga_tuner_exp_1-2_wrapper.sbatch"
batch_path = os.path.join( batch_dir, file_name )
with open( batch_path, 'w' ) as f:
    f.write( slurm_batch_wrapper_txt )


# ## Run the SLURM Script
# 
# Try running this cmd line to launch the SLURM script
# 
# ```
# sbatch /home/mbo2d/galStuff/slurm_batch_scripts/spam_ga_tuner_exp_1-2_wrapper.sbatch 62000000190717 balance_base test /home/mbo2d/galStuff/data/spam_tuner_exp_1-2
# ```
# 
# You should see output like this. `Submitted batch job 44879`
# Use `squeue -u mbo2d` to see your job in the queue!
# And check folder `/home/mbo2d/galStuff/slurm_batch_logs/` for a new log file!
# NOTE: My logger outputs everything to the .err file so check that file for output!
# ```
# [2026-01-09 14:55:26,482 - INFO - compare_tng_to_spam.func_experiment_1]: Running Experiment 1
# ...
# [2026-01-09 14:55:27,894 - INFO - optimization_manager.__init__]: Initializing PYGAD Manager
# ...
# [2026-01-09 14:55:28,966 - INFO - parallel_queue_manager._run_local_processes]: Local Manager: Progress: [==                  ] 3/30
# [2026-01-09 14:55:29,558 - INFO - parallel_queue_manager._run_local_processes]: Local Manager: Progress: [====================] 30/30 - Queue Complete
# [2026-01-09 14:55:30,386 - INFO - parallel_queue_manager._run_local_processes]: Local Manager: Progress: [====================] 25/25 - Queue Complete
# ...
# [2026-01-09 14:55:37,768 - INFO - compare_tng_to_spam.func_experiment_1]: Results saved to /home/mbo2d/galStuff/data/spam_tuner_exp_1/Exp0_target=62000000190717__score=balance_base__ga=test_results.png
# 
# ```
# 
# ---
# 
# 
# Next!  Let's create a test Python scrip to run 4 targets in test mode to verify everything is working!  See code below that creates that script!....  NVM, apparently i'm missing most of those particles... Top 10!

# In[ ]:


# Get list of available particles in data file

# list all files in dir
data_dir = '/home/mbo2d/galStuff/data/tng-data'
import os
files = os.listdir(data_dir)

# Filter to files that contain 'target-archive' in name
files = [ f for f in files if 'target-archive' in f]

# only grab the target id in the name
top_ids = [ f.split('-')[2].split('.')[0] for f in files ]
# print(top_ids)


# In[ ]:


test_job_submit_txt = """import os, subprocess

top_id_file = '/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-4-top-ratings.txt'
sbatch_script = '/home/mbo2d/galStuff/slurm_batch_scripts/spam_ga_tuner_exp_1-2_wrapper.sbatch'
output_dir = '/home/mbo2d/galStuff/data/spam_tuner_exp_1-2'

score = 'balance_base'
ga = 'test'

# Let's try this.
data_dir = '/home/mbo2d/galStuff/data/tng-data'
files = os.listdir(data_dir)

# Filter to files that contain 'target-archive' in name
files = [ f for f in files if 'target-archive' in f]

# only grab the target id in the name
top_ids = [ f.split('-')[2].split('.')[0] for f in files ]

# Take first X
for tid in top_ids[:4]:
    cmd = [
        'sbatch',
        sbatch_script,
        str(tid),
        score,
        ga,
        output_dir,
    ]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)
"""

batch_dir = '/home/mbo2d/galStuff/slurm_batch_scripts/'

# create dir if it doesn't exist
os.makedirs( batch_dir, exist_ok=True )


# Let's save the batch script
file_name = "spam_ga_tuner_exp_1-2_test_job_submit.py"
batch_path = os.path.join( batch_dir, file_name )
with open( batch_path, 'w' ) as f:
    f.write( test_job_submit_txt )


# Run 
# ```
# python3 ./slurm_batch_scripts/spam_ga_tuner_exp_1-2_test_job_submit.py
# ```
# and you'll see something like this!
# ```
# [mbo2d@c2 galStuff]$ python3 ./slurm_batch_scripts/spam_ga_tuner_exp_1-2_test_job_submit.py 
# Submitting: sbatch /home/mbo2d/galStuff/slurm_batch_scripts/spam_ga_tuner_exp_1-2_wrapper.sbatch 51000000196622 balance_base balance /home/mbo2d/galStuff/data/spam_tuner_exp_1-2
# Submitted batch job 44892
# Submitting: sbatch /home/mbo2d/galStuff/slurm_batch_scripts/spam_ga_tuner_exp_1-2_wrapper.sbatch 54000000045003 balance_base balance /home/mbo2d/galStuff/data/spam_tuner_exp_1-2
# Submitted batch job 44893
# Submitting: sbatch /home/mbo2d/galStuff/slurm_batch_scripts/spam_ga_tuner_exp_1-2_wrapper.sbatch 49000000112167 balance_base balance /home/mbo2d/galStuff/data/spam_tuner_exp_1-2
# Submitted batch job 44894
# Submitting: sbatch /home/mbo2d/galStuff/slurm_batch_scripts/spam_ga_tuner_exp_1-2_wrapper.sbatch 37000000126111 balance_base balance /home/mbo2d/galStuff/data/spam_tuner_exp_1-2
# Submitted batch job 44895
# ```

# In[ ]:





# In[16]:


test2_submit_txt = """import os, subprocess

top_id_file = '/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-4-top-ratings.txt'
sbatch_script = '/home/mbo2d/galStuff/slurm_batch_scripts/spam_ga_tuner_exp_1-2_wrapper.sbatch'
output_dir = '/home/mbo2d/galStuff/data/spam_tuner_exp_1-2'

score = 'balance_base'
ga_list = [ 'balance', 'explore', 'exploit' ]

# Let's try this.
data_dir = '/home/mbo2d/galStuff/data/tng-data'
files = os.listdir(data_dir)

# Filter to files that contain 'target-archive' in name
files = [ f for f in files if 'target-archive' in f]

# only grab the target id in the name
top_ids = [ f.split('-')[2].split('.')[0] for f in files ]
tid = 62000000190717

# Take first X
for ga in ga_list:
    cmd = [
        'sbatch',
        sbatch_script,
        str(tid),
        score,
        ga,
        output_dir,
    ]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)
"""

batch_dir = '/home/mbo2d/galStuff/slurm_batch_scripts/'

# create dir if it doesn't exist
os.makedirs( batch_dir, exist_ok=True )


# Let's save the batch script
file_name = "spam_ga_tuner_exp_1-2_test2_job_submit.py"
batch_path = os.path.join( batch_dir, file_name )
with open( batch_path, 'w' ) as f:
    f.write( test2_submit_txt )


# In[20]:


main_job_submit_txt = """import os, subprocess

top_id_file = '/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-4-top-ratings.txt'
sbatch_script = '/home/mbo2d/galStuff/slurm_batch_scripts/spam_ga_tuner_exp_1-2_wrapper.sbatch'
output_dir = '/home/mbo2d/galStuff/data/spam_tuner_exp_1-2'

score = 'balance_base'
ga_list = [ 'balance', 'explore', 'exploit' ]

# Let's try this.
data_dir = '/home/mbo2d/galStuff/data/tng-data'
files = os.listdir(data_dir)

# Filter to files that contain 'target-archive' in name
files = [ f for f in files if 'target-archive' in f]

# only grab the target id in the name
top_ids = [ f.split('-')[2].split('.')[0] for f in files ]

for tid in top_ids:
    for ga in ga_list:
        cmd = [
            'sbatch',
            sbatch_script,
            str(tid),
            score,
            ga,
            output_dir,
        ]

        print("Submitting:", " ".join(cmd))
        subprocess.run(cmd, check=True)
"""

batch_dir = '/home/mbo2d/galStuff/slurm_batch_scripts/'

# create dir if it doesn't exist
os.makedirs( batch_dir, exist_ok=True )


# Let's save the batch script
file_name = "spam_ga_tuner_exp_1-2_main_job_submit.py"
batch_path = os.path.join( batch_dir, file_name )
with open( batch_path, 'w' ) as f:
    f.write( main_job_submit_txt )


# In[ ]:





# In[ ]:




