#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# coding: utf-8

"""
File: spam_tuner_ga.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2024-Mar-12

References:  Sections of this code were enhanced with the assistance of ChatGPT made by OpenAI. 

Description:  This script is to fine-tune the SPAM hyperparameters in order for to produce SPAM orbits
    that produce tidal features similar to those from the TNG orbits.
    
"""

# ================================ STANDARD IMPORTS ================================ #
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================ PROJECT MODULES ================================ #
# Add main project directory to import project modules
try:
    SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
    sys.path.append(PROJECT_DIRECTORY)

except:
    # Assume we are running from the script directory
    SCRIPT_DIRECTORY = os.getcwd()
    PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
    
    # Also assume you might be in galStuff
    work_dir = os.getcwd()
    if 'galStuff' == work_dir.split('/')[-1]:
        PROJECT_DIRECTORY =  work_dir + '/galaxyJSPAM'
        sys.path.append(PROJECT_DIRECTORY)

# import project modules
import utilities.general_utility as gu
import utilities.parallel_queue_manager as qm
import Optimization_Methods.optimization_manager as om
import IllustrisTNG.tng_functions as tf
import IllustrisTNG.convert_tng_to_spam as cvt_ts
import IllustrisTNG.compare_tng_to_spam as cmp_ts

# ================================ GLOBAL VARIABLES ================================ #
buildEnv = False

# Global logger from main program
logger = gu.configure_logger()


# In[ ]:





# In[ ]:


if buildEnv:
    p1 = 62000000190717
    p2 = 75000000121160
    spam_loc = "/home/mbo2d/galStuff/data/misc_spam_orbits"

    logger.setLevel(logging.INFO)

    t1 = spam_tuner.DEV_SPAM_TNG_Tuner( p1, )

    orbit = t1.simulate_orbit( orbit_data={}, save_dir=spam_loc )
    weight_func = t1.gen_orbit_weights( boundary_value=0.1, scale_past=0.5 )
    
    print(f"Orbit: {type(orbit)}")
    print(f"Orbit: {[type(orbit[i]) for i in range(len(orbit))]}")
    print(f"Orbit: {orbit[0].shape}")
    print(f"Orbit: {orbit[1].keys()}")

    print(f"Weight Func: {type(weight_func)}")
    score = t1.weighted_avg_error_sq_over_by_time( orbit[0], weight_func )
    print(f"Score: {score}")


# In[ ]:


if buildEnv:

    # I want to see what targets I have available to compare to.
    pts_loc = "/home/mbo2d/galStuff/data/tng-data/"
    target_loc = ""


# In[ ]:


if buildEnv:
    reload(gu)
    reload(om)
    reload(qm)
    reload(spam_tuner)
    logger = gu.configure_logger( log_level=logging.WARNING, logger_name='test_tuner')

    spam_hyper_param = [
        {'name':'lnl', 'type': 'float', 'bounds': [1e-3, 2e-2], 'prior':'log-uniform', 'transform':'normalize',  },
        {'name':'m_halo', 'type': 'float', 'bounds': [1, 18], 'prior':'uniform', 'transform':'normalize', },
        {'name':'r_scale', 'type': 'float', 'bounds': [10, 30], 'prior':'uniform', 'transform':'normalize', },
    ]

    space_manager = om.Space_Manager( spam_hyper_param )
    r_inputs = space_manager.get_random_sample( 10 )
    print( f"Random Inputs: \n{r_inputs}" )

    def exec_wrapper( x ):
        spam_hparam = { 'lnl': x[0], 'm_halo': x[1], 'r_scale': x[2] }
        orbit = t1.simulate_orbit( orbit_data=spam_hparam, save_dir=spam_loc )
        orbit_error = t1.weighted_avg_error_sq_over_by_time( orbit[0], weight_func )
        return orbit_error

    exec_manager = om.Execution_Manager( space_manager=space_manager, 
                                        black_box_function=exec_wrapper, 
                                        normalize_output='reciprocal',
                                        func_logger='spam_tuner',
    )

    r_outputs = exec_manager.evaluate_function( r_inputs )
    print( f"Random Outputs: \n{r_outputs}" )
    


# In[ ]:


if buildEnv:
    
    #####   GA MANAGER   ######
    ga_config = {
        "num_generations": 5,      # Number of generations to run the GA.
        "num_parents_mating": 4,  # Number of solutions to be selected as parents.
        "sol_per_pop": 10,        # Number of solutions per population.
        "gene_type": float,        # Data type of the genes. Can be float, int, or a list of types.
        "keep_parents": 2,         # Number of top solutions to keep in the next generation.
        "parent_selection_type": "sus",   # Parent selection strategy 
        "crossover_type": "uniform", # Crossover strategy.        
        "mutation_probability": 0.5,      # Probability of mutating a gene.
        "random_mutation_min_val": -0.1,  # Minimum value for random mutation.
        "random_mutation_max_val":  0.1,  # Maximum value for random mutation.
    }

    ga_manager = om.PYGAD_Manager( ga_config,  
                                  space_manager = space_manager,
                                  execution_manager = exec_manager,                                  
                                  )
    
    ga_manager.gad.run()


# In[11]:


if buildEnv:
    
    # Create subplot for each dimension
    fig, axs = plt.subplots(3, figsize=(16, 12))

    # Set main title
    fig.suptitle("Evaluation Manager Random Results")

    # Histogram of inputs and outputs
    for i, dim_info in enumerate(space_manager.space_info):
        # inputs
        exec_manager.scatter_input(dim_index=i, space='original', external_ax=axs[i], title=f"Orig In vs Out: {dim_info['name']}", reverse_color=True)


# In[ ]:




