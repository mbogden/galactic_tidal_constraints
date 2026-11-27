#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# coding: utf-8

"""
File: misc_testing.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2025-Mar-17

References:  Sections of this code were enhanced with the assistance of ChatGPT made by OpenAI.
"""

# ================================ STANDARD IMPORTS ================================ #
import logging, os

# partial imports
from mpi4py import MPI

# ================================ PROJECT MODULES ================================ #
# Add main project directory to import project modules
try:
    SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
    sys.path.append(PROJECT_DIRECTORY)

except:
    # Consider we are running from the script directory
    SCRIPT_DIRECTORY = os.getcwd()
    PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
    
    # Also consider you might be in galStuff directory
    work_dir = os.getcwd()
    if 'galStuff' == work_dir.split('/')[-1]:
        PROJECT_DIRECTORY =  work_dir + '/galaxyJSPAM'
        sys.path.append(PROJECT_DIRECTORY) 

# import project modules
import utilities.general_utility as gu

# ================================ GLOBAL VARIABLES ================================ #
buildEnv = False

# Global logger from main program
SCRIPTS_ARGUMENTS, logger = gu.initialize_environment( module_name ='testing' ) # Testing using multiple loggers for tiered log levels.
logger.info("Imported Loaded")


# In[ ]:


class Test_Class:

    def __init__( self, foo ):
        self.logger = gu.configure_logger( logger_name = 'Test_Class')
        self.logger.info("Testing Test Class")
    
    def test_log( self, ):
        self.logger.info("Testing Log")

c = Test_Class( 12 )


# In[ ]:




