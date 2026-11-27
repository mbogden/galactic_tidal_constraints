#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author:	 Matthew Ogden
    Created:	01 Sep 2020
Description:	Hopefully my primary code for calling all things SPAM Simulation, Imaging, and Genetic Algorithm.
'''

# Python module imports
from os import path, listdir, mkdir
from sys import path as sysPath, exit
from copy import deepcopy
from shutil import rmtree

import pandas as pd 
import numpy as np 
import cv2 
from time import sleep 
from mpi4py import MPI
from mpi_master_slave import Master, Slave, WorkQueue

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# For loading in Matt's general purpose python libraries
import Support_Code.general_module as gm
import Support_Code.info_module as im
import Simulator.main_simulator as SIM
import Image_Creator.main_image_creator as IMG
import Feature_Extraction.main_feature_extraction as fe
import Neural_Network.main_neural_networks as nn
sysPath.append( path.abspath( 'Machine_Score/' ) )
from Machine_Score import main_machine_score as ms
from Genetic_Algorithm import main_Genetic_Algorithm as ga
import Score_Analysis.main_score_analysis as sa

def test():
    print("SIMR: Hi!  You're in Matthew's main program for all things galaxy collisions")

# For testing and developement
new_func = None
def set_new_func( inFunc ):
    global new_func
    new_func = inFunc
# End set new function

def main(arg):

    if arg.printBase and mpi_rank == 0:
        test()

    if mpi_size > 1:
        sleep(mpi_rank*0.015)
        if mpi_rank == 0 and arg.printBase:
            print( 'SIMR: main: In MPI environment!')
        if arg.printBase: gm.tabprint('I am %d / %d on %s' %( mpi_rank, mpi_size, MPI.Get_processor_name() ) ) 
        mpi_comm.Barrier()
        
    if arg.printAll and mpi_rank == 0:
        arg.printArg()
        gm.test()
        SIM.test()
        IMG.test()
        fe.test()
        nn.test()
        ms.test()

    # end main print

    if arg.simple:
        if arg.printBase and mpi_rank == 0: 
            print("SIMR: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None and mpi_rank == 0:
        run_new_score( arg )

    elif arg.targetDir != None:
        target_main( arg )

    elif arg.dataDir != None:
        Multi_Target( arg )
    
    elif mpi_rank == 0:
        print("SIMR: main: Nothing selected!")
        print("SIMR: main: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")
        print("\t - targetDir /path/to/dir/")
        print("\t - dataDir /path/to/dir/")
    
# End main

def Multi_Target( cmdArg = gm.inArgClass() ):
    
    if mpi_rank == 0:
        print("SIMR: Multi_Target:")
    dDir = cmdArg.get( 'dataDir', None )
    dataDir = gm.validPath( dDir )
    
    if dataDir == None:
        if mpi_rank == 0:
            print("WARNING: SIMR.Multi_Target:")
        gm.tabprint("Invalid dataDir: %s" % dDir)
        return None
    
    else:
        if mpi_rank == 0:
            gm.tabprint('dataDir: %s' % dataDir)
    
    tNames = listdir( dataDir )
    tNames.sort()
    
    for n in tNames:    
        
        if mpi_rank == 0:  print('****************************')
        
        # Prep variables
        tDir = dataDir + n
        tArg = deepcopy(cmdArg)
        tArg.setArg('targetDir', tDir)
        
        target_main( tArg )
        
    if mpi_rank == 0:  print('****************************')
    
# End Multi_Target


# Process target directory
def target_main( cmdArg=gm.inArgClass(), tInfo = None ):
    
    printAll = cmdArg.printAll
    printBase = cmdArg.printBase
    
    if cmdArg.printAll:
        print("SIMR: target_main:")

    # Initialize target_info from disk and cmd arguments
    if mpi_rank == 0 and tInfo == None:
        tInfo = target_initialize( cmdArg, tInfo )
        
    # If in mpi_env, send to others
    if mpi_size > 1:
        
        if mpi_rank == 0:
            if printBase: print("SIMR: target_main: sending tInfo to others")
            tInfo = mpi_comm.bcast( tInfo, root=0 )

        # others waiting for target info class. 
        else:
            tInfo = None
            tInfo = mpi_comm.bcast( tInfo, root=0 )

    # Have all check for valid tInfo
    if tInfo == None or tInfo.status == False:
        return None
    
    # If performing new Genetic Algorithm Experiment
    if cmdArg.get( 'gaExp', False ):
        GA_Experiment_Wrapper( cmdArg, tInfo )
    
    # For testing new score workers
    elif cmdArg.get( 'newGen', False ):
        
        # Default should be used in mpi_env
        if mpi_size > 1:
            target_test_new_gen_scores( cmdArg, tInfo )
        
        else:
            target_test_new_gen_scores_single( cmdArg, tInfo )
    
    # For creating new orbits
    if cmdArg.get( 'newOrbit', False ):
        pass
        #target_new_orbits( tInfo, cmdArg )
    
    # Create new files/scores if called upon
    if cmdArg.get( 'newAll', False ) \
    or cmdArg.get( 'newTargetImage', False ) \
    or cmdArg.get( 'newSim', False ) \
    or cmdArg.get( 'newImage', False ) \
    or cmdArg.get( 'newFeats', False ) \
    or cmdArg.get( 'newScore', False ) \
    or cmdArg.get( 'normFeats', False ):            
        target_new_scores( tInfo, cmdArg )
    
    else:
        if printBase and mpi_rank == 0: 
            print("SIMR.simr_target:  Nothing Selected.")
    
def GA_Experiment_Wrapper( cmdArgs, tInfo ):
    
    ##############################################
    #####   Variable and Parameter Setup    ###### 
    ##############################################
    
    printBase = cmdArgs.printBase
    printAll = cmdArgs.printAll
    runArgs = None
    
    if mpi_rank == 0:
        
        if printBase:
            print( "SIMR.GA_Experiment_Wrapper:" )
            gm.tabprint("Setting up parameters")
        
        # Prepare score parameters for new models
        prep_score_parameters( cmdArgs, tInfo )
        
        # Create and edit cmdArgs for model runs
        runArgs = target_prep_cmd_params( tInfo, cmdArgs )
        runArgs.setArg( 'newInfo', True )
        runArgs.setArg( 'newAll', True )

        # Validate GA parameters
        ga_param = cmdArgs.get('gaParam', None)
        if cmdArgs.get('gaParam', None) == None and cmdArgs.get('gaParamLoc', None) == None:
            print("WARNING: SIMR.GA_Experiment_Wrapper:")
            gm.tabprint("Please provide Genet. Algorithm Parameters")
            gm.tabprint("\"-gaParamLoc path/to/ga_param.json\"")
            runArgs = None
        
        elif cmdArgs.get('gaParam', None) == None and cmdArgs.get('gaParamLoc', None) != None:
            gaPLoc = cmdArgs.get('gaParamLoc', None)
            ga_param = ga.Prep_GA_Input_Parameters( tInfo, gaPLoc )
        
        # If invalid ga_param, modify runArgs so workers know to exit too.
        if ga_param == None:
            runArgs = None
        
        # Prepare for results
        tmpDir = tInfo.get('tmpDir')    # This might be changed later... 
        outDir = gm.validPath( tmpDir )
        
        if outDir != None and ga_param != None:
            # create details file for ga_results
            rName = '%s_%s' % ( ga_param.get('name', 'Testing'), gm.getFileFriendlyDateTime() )
            resultsLocBase = outDir + rName + '_'
            
            print("SIMR.GA_Experiment_Wrapper:")
            gm.tabprint("Saving results: %s" % resultsLocBase) 
            
            ga_details = {}
            ga_details['name'] = rName
            ga_details['target_id'] = tInfo.get('target_id')
            ga_details['info'] = 'This file is the prototype for creating \
                the Genetic Algorythm pipeline.'
            ga_details['score_parameters'] = cmdArgs.get('scoreParams')
            ga_details['ga_paramters'] = ga_param
            
            detailsLoc = resultsLocBase + 'details.json'
            gm.saveJson( ga_details, detailsLoc, pretty = True, convert_numpy_array=True )
    
        else:
            print("WARNING: SIMR.GA_Experiment_Wrapper:")
            gm.tabprint("Results directory not found: %s" % tmpDir ) 
            runArgs = None  # Set runArgs to None so workers no to exit too.
            
    
    # Have rank 0 broadcast model run arguments
    if mpi_rank == 0:
        mpi_comm.bcast( runArgs, root=0 )

    # If in mpi and expecting info.
    else:
        runArgs = mpi_comm.bcast( None, root=0 )

    mpi_comm.Barrier()
    
    # Exit program if invalid arguments or parameters. 
    if type(runArgs) == type(None):
        if mpi_rank == 0: 
            print("WARNING: SIMR.GA_Experiment_Wrapper: Invalid run arguments")
        return None
    
    # End parameter preperation
      
    ####################################
    #####   Single Core Testing   ###### 
    ####################################
    
    if mpi_size == 1:
        
        # Prepare scoring class/function 
        scorer = ga_simple_score_wrapper( runArgs, tInfo, ga_param['parameter_psi'], True )
        
        # Call Genetic Algorithm Experiment
        ga.Genetic_Algorithm_Experiment( ga_param, scorer.score_models, resultsLocBase, True )
        
        return
        
    # End single core testing
    
    #####################################
    #####   MPI GA Implementaion   ######
    #####################################
    
    # If not rank 0, create Workers and have them on standby for score creation
    if mpi_rank != 0:
        
        # This creates the workers and they'll remain here on standby to score models.
        # Master will utilize and terminate workers elsewhere in code.
        new_score_worker(tInfo, runArgs).run()
        print("Worker %d:  Terminated" % mpi_rank)
        return
        
    # Should only reach this point in code if Master (rank == 0)
    
    # Create wrapper class to use mpi queue and workers to score models
    mpi_queue_wrapper = ga_mpi_score_wrapper( ga_param['parameter_psi'], printAll )
    
    # Call Genetic Algorithm Experiment
    ga.Genetic_Algorithm_Experiment( ga_param, mpi_queue_wrapper.score_models, resultsLocBase, True )
    
    # Experiment done, terminate workers
    mpi_queue_wrapper.terminate_workers()

# END GA_Experiment_Wrapper

class ga_mpi_score_wrapper:
    
    psi     = None # variable needed for converting from ga to spam parameters
    prog    = None # variable for printing scoring progress
    
    master    = None # Master for mpi master worker 
    mpi_queue = None # Queue the workers will pull model data from.
    
    def __init__( self, psi, printProg, sleepTime = 0.1 ):
        
        # Save input variables
        self.psi     = psi
        self.prog    = printProg
        self.sleepTime = sleepTime
        
        # Create variables for Master/Worker and Queue
        self.master = Master( range(1, mpi_size ) )
        self.mpi_queue = WorkQueue( self.master )
        
        print("SIMR.ga_mpi_score_wrapper: Master Created")
        
    def score_models( self, ga_params ):

        if self.prog: print("SIMR.ga_mpi_score_wrapper.score_models:")
        
        # Useful variables
        n = ga_params.shape[0]
        scores = np.zeros(n)
        c = 0
        
        # Convert incoming ga parameters to spam parameters
        spam_parameters = ga.convert_ga_to_spam( ga_params, self.psi )
        
        # Add model data to mpi queue
        for i in range( n ):
            self.mpi_queue.add_work( data=( i, spam_parameters[i,:] ) )
            
        # Stay in loop until queue is empty
        while not self.mpi_queue.done():
            
            # Check if a worker(s) is available, do work.
            self.mpi_queue.do_work()  # Linked to class new_score_worker somehow
            
            # Sleep for a time and wait for workers to return data. 
            sleep( self.sleepTime )

            # Get return data if available
            for i, score in self.mpi_queue.get_completed_work():
                c += 1

                # Save data
                if score != None:  scores[i] = score

                # Print progress
                if self.prog:  print("mpi_queue: %4d / %4d: %d - %s" % ( c, n, i, str(score) ), end='\r' )
            
            # End processing return data
            
        # End, mpi queue is now empty
        
        print("\nSIMR.score_models_worker_queue: %4d / %4d - Complete\n" % ( n, n ) )
        
        return scores
    
    def terminate_workers( self, ):
        self.master.terminate_slaves()
        
# End ga_mpi_score_wrapper
    
def target_test_new_gen_scores_single( cmdArgs, tInfo ):    
    
    printBase = tInfo.printBase
    printAll = tInfo.printAll
    
    if printBase:
        print( "SIMR.target_test_new_gen_scores_single: %d of %d" %(mpi_rank,mpi_size) )

    # Use function to prepare tInfo and score params
    runArgs = target_prep_cmd_params( tInfo, cmdArgs )

    # Add args for new score and info for each run
    if type( runArgs ) != type( None ):
        runArgs.setArg( 'newAll', True )
        runArgs.setArg( 'newInfo', True )
    
    # Have all verify if they have valid arguments
    if type(runArgs) == type(None):
        print("WARNING %d: SIMR.target_test_new_gen_scores: Invalid run arguments" % mpi_rank)
        return None
    

    gm.tabprint("Using zoo merger data for testing")
    
    # WORKING: Use zoo merger data for testing and developing.
    zScores, mData = tInfo.getOrbParam()
    scores = score_models_iter( tInfo, runArgs, mData[0:7,:] )
    print(scores)

    pass


class ga_simple_score_wrapper:
    
    runArgs = None # arguments while scoring individual models
    tInfo   = None # class for target system
    psi     = None # variable needed for converting from ga to spam parameters
    prog    = None # variable for printing scoring progress
    
    def __init__( self, runArgs, tInfo, psi, printProg ):
        
        # Save variables
        self.runArgs = runArgs
        self.tInfo   = tInfo
        self.psi     = psi
        self.prog    = printProg
        
        if self.prog:  print("SIMR.ga_simple_score_wrapper")
        
    def score_models( self, ga_params ):
        spam_parameters = ga.convert_ga_to_spam( ga_params, self.psi )
        scores = score_models_iter( self.tInfo, self.runArgs, \
                                        spam_parameters, printProg = self.prog )
        return scores

# End ga_machine score wrapper

def score_models_iter( tInfo, runArgs, newData, printProg = True ):    
    
    printBase = tInfo.printBase
    n = newData.shape[0]
    scores = np.zeros(n)
    
    worker = new_score_worker(tInfo, runArgs) 
    
    if printProg:  print('Master: Starting %d models' % n ) 
    for i in range( n ):
        ir, score = worker.do_work( (i, newData[i,:]) )
        scores[ir] = score
        if printProg:  print('Master: Received %d / %d - %s' % ( ir+1, n, str(score) ), end='\r' ) 
    if printProg:  print("\nMaster: Complete")
        
    return scores

def target_test_new_gen_scores( cmdArgs, tInfo ):
    
    ###################################
    #####   Worker Preperation   ######
    ###################################
    
    printBase = tInfo.printBase
    printAll = tInfo.printAll
    
    if printBase:
        print( "SIMR.target_test_new_gen_scores: %d of %d" %(mpi_rank,mpi_size) )
      
    # Make sure you're in an MPI_environment with more than 1 core.
    if mpi_size == 1:
        print("WARNING: SIMR.target_test_new_gen_scores:")
        gm.tabprint("Must run code in mpirun with more than 1 core to work properly")
        gm.tabprint("Example: 'mpirun -n 4 python3 main_simr.py -gaExp'")
        return
    
    # Prepare arguments for runs
    runArgs = None

    # Have rank 0 prep args and broadcast
    if mpi_rank == 0:
        
        # Use function to prepare tInfo and score params
        runArgs = target_prep_cmd_params( tInfo, cmdArgs )
        
        # Add args for new score and info for each run
        if type( runArgs ) != type( None ):
            runArgs.setArg( 'newAll', True )
            runArgs.setArg( 'newInfo', True )
        
        # Broadcast master runArgs to everyone.
        mpi_comm.bcast( runArgs, root=0 )

    else:
        # Workers expecting runArgs.
        runArgs = mpi_comm.bcast( runArgs, root=0 )
    
    # Everyone wait
    mpi_comm.Barrier()
    
    # Have all verify if they have valid arguments
    if type(runArgs) == type(None):
        print("WARNING %d: SIMR.target_test_new_gen_scores: Invalid run arguments" % mpi_rank)
        return None
    
    #################################
    #####   Initiate Workers   ######
    #################################
    
    # Create Workers and have them on standby to create scores
    if mpi_rank != 0:
        new_score_worker(tInfo, runArgs).run()
        print("Worker %d:  Terminated" % mpi_rank)
        return
        
    ########################################
    #####   Testing new generations   ######
    ########################################
    
    zScores, mData = (None, None)

    gm.tabprint("Using zoo merger data for testing")
    # Create Master Class.
    master = Master( range(1, mpi_size ) )
    
    # Create Queue, all workers will be using this
    mpi_queue = WorkQueue( master )

    # Send data to receive scores
    
    # WORKING: Use zoo merger data for testing and developing.
    zScores, mData = tInfo.getOrbParam()
    print("Model Data", mData.shape)
    
    print("Master 0: Sending first 10 items")
    newScores = score_models_worker_queue( mpi_queue, mData[0:10,:], printBase, printAll )
    print("Master 0: First Scores", newScores)

    # Create second queue of items for testing.
    print("Master 0: Sending second 15 items")
    newScores = score_models_worker_queue( mpi_queue, mData[10:25,:], printBase, printAll )
    print("Master 0: Second Scores", newScores)

    # Create Third queue of items.
    print("Master 0: Sending third 25 items")
    newScores = score_models_worker_queue( mpi_queue, mData[25:50,:], printBase, printAll )
    print("Master 0: Third Scores", newScores)

    print("Master 0: Terminating Slaves.")
    master.terminate_slaves()
    
# End target_test_new_gen_scores


def score_models_worker_queue( mpi_queue, newData, printBase = True, printAll = False ):
    
    if printBase: print("SIMR.score_models_worker_queue:")
        
    # Useful variables
    n = newData.shape[0]
    c = 0
    scores = np.zeros(n)

    
    # Create queue of items
    for i in range( n ):
        mpi_queue.add_work( data=( i, newData[i,:] ) )

    # Stay in loop until queue has items
    while not mpi_queue.done():

        # If workers are available, tell worker to do work.
        mpi_queue.do_work()  # Linked to class new_score_worker

        # Get return data if available
        for i, score in mpi_queue.get_completed_work():
            
            # Save data
            c += 1
            if score != None:
                scores[i] = score
            
            # Print progress
            if printAll:
                print("Master received: %d - %s" % ( i, score ) )
                
            if printBase:
                print("SIMR.score_models_worker_queue: %4d / %4d: %d - %s" % ( c, n, i, str(score) ), end='\r' )
        
    # End while not mpi_queue.done()
    print("\nSIMR.score_models_worker_queue: %4d / %4d - Complete\n" % ( n, n ) )
    
    return scores


class new_score_worker(Slave):
    
    rank = mpi_rank
    worker_dir = None
    printWorker = None
    rmRunDir = True

    def __init__(self, tInfo, runArgs ):
        
        super(new_score_worker, self).__init__()
        
        # Save info for creating scores later. 
        self.tInfo = tInfo
        self.runArgs = runArgs
        self.printWorker = runArgs.get( 'printWorker', False )
        self.rmRunDir = runArgs.get( 'rmRunDir', True )
        self.score_names = [ name for name in self.runArgs.scoreParams ]
        
        if self.printWorker or self.runArgs.printBase:  
            print("Worker %d: __init__:" % mpi_rank )
        
        # Where to save new run data 
        if self.runArgs.get( 'workerLocName', 'target' ) == 'target':
            
            tmpDir = self.tInfo.get('tmpDir', None)
            if tmpDir == None:
                print("%d WARNING: new_score_worker.__init__: tmpDir not found")
            
            else:
                
                self.worker_dir = tmpDir + 'worker_dir_%s/' % str(self.rank).zfill(4)
                
                if gm.validPath( self.worker_dir ) == None:
                    print("%d creating dir: %s" % (self.rank, self.worker_dir))
                    mkdir( self.worker_dir )
            
                    
        # For Babbage, our local cluster
        elif self.runArgs.get( 'workerLocName', None ) == 'babbage':
            
            # Hardcoded location of the tmp dir for babbages local computers
            tmpBabbageDir = '/state/partition1/'
            #tmpBabbageDir = 'tmpDir/'
            
            # If invalid location, complain
            if gm.validPath( tmpBabbageDir ) == None:
                self.worker_dir = None
                print("WARNING %d: Invalid dir: %s" % ( mpi_rank, tmpBabbageDir ) )
            
            # Working local diskspace is valid
            else:
                
                # Hardcoded
                tmpDir = tmpBabbageDir + 'mbo2d_simr/'
                
                if gm.validPath( tmpDir ) == None:
                    
                    # Wait to create, I'm hoping to avoid multiple people creating it at the same time.
                    waitTime = np.random.uniform() * 10
                    sleep(waitTime)
                    
                    # Hoping only one person will create and everyone else 
                    if gm.validPath( tmpDir ) == None:
                        mkdir( tmpDir )
                        print("Worker %d created tmp dir: %s" % (mpi_rank, tmpDir ) )
                    
                # Everyone meet up again
                if gm.validPath( tmpDir ) != None:
                    
                    self.worker_dir = tmpDir + 'worker_dir_%s/' % str(self.rank).zfill(4)
                    
                    # Check if worker directory already exists
                    if gm.validPath( self.worker_dir ) == None: 
                        print("%d creating dir: %s" % (self.rank, self.worker_dir))
                        mkdir( self.worker_dir )
                
                # Check if valid, if not create.
                    
        # Final command to ensure it's either a valid location or None.
        self.worker_dir = gm.validPath( self.worker_dir )

    # Function for creating a machine score out of SPAM parameters.
    def do_work( self, data ):
        
        # Extract needed data to run
        i, mData = data
        
        if self.printWorker:
            print('Worker %d: do_work: received %s' % (mpi_rank, i), )
            
        # Double check if Worker has a valid working directory
        if type( self.worker_dir ) == type( None ):
            return (i, None)
        
        # Create run variables
        run_id = 'run_%s' % str(i).zfill(4)
        runDir = self.worker_dir + '%s/' % run_id
        infoLoc = runDir + 'base_info.json'
        model_string = ','.join( map(str, mData) )
        
        # Create the information file for the run/model
        mInfo = {}
        mInfo['run_id'] = run_id
        mInfo['model_data'] = model_string
        
        # Create model directories and info file
        if gm.validPath( runDir ) == None:
            mkdir( runDir )
            
        gm.saveJson( mInfo, infoLoc )
        
        # Create rInfo class to verify working directory and info file
        rInfo = im.run_info_class( runDir = runDir, rArg = self.runArgs )
        rInfo.tInfo = self.tInfo
        
        # Return if bad. 
        if rInfo.status == False:
            return (i, None)
        
        # Create simulation, image and compute score
        run_new_score( cmdArg = self.runArgs, rInfo = rInfo )
        
        # Built to create several scores, returning only first atm
        scores = []
        for name in self.score_names:
            scores.append( rInfo.getScore( name ) )
            
        # Remove directory to convserve disk space.
        if self.rmRunDir:  rmtree( runDir )
        
        if self.printWorker:
            print('Worker %d: do_work: Complete: %d - %s' % (mpi_rank, i, str(scores[0])), )
        
        return (i, scores[0])
    
# End class new_score_worker

            
        
def target_initialize( cmdArg=gm.inArgClass(), tInfo = None ):
    
    tDir = cmdArg.targetDir
    printBase = cmdArg.printBase
    printAll = cmdArg.printAll
    
    if cmdArg.printAll: 
        cmdArg.printBase = True

    if printBase:
        print("SIMR: target_initialize:")
        gm.tabprint("tDir: %s" % tDir )
        gm.tabprint("tInfo: %s" % type(tInfo) )

    # Check if given a target
    if tInfo == None and tDir == None and cmdArg.get('tInfo') == None:
        gm.eprint("WARNING: SIMR.target_initialize:")
        gm.etabprint("Please provide either target directory or target_info_class")
        return

    # Read target directory if location given. 
    elif tInfo == None and tDir != None:
        
        tInfo = im.target_info_class( targetDir=tDir, tArg = cmdArg )

        # If creating a new base, create new images
        if cmdArg.get('newBase',False):
            try:
                chime_0 = tInfo.readScoreParam( 'chime_0' )
                chime_image = IMG.adjustTargetImage( tInfo, chime_0['chime_0'], \
                                                   printAll = cmdArg.printAll )
                tInfo.saveWndchrmImage( chime_image, chime_0['chime_0']['imgArg']['name'] )
            except: pass

        # Gather scores if called for
        if cmdArg.get('update',False):
            tInfo.gatherRunInfoFiles()
            tInfo.updateScores()
            tInfo.saveInfoFile()

    # get target if in arguments. 
    elif tInfo == None and cmdArg.tInfo != None:
        tInfo = cmdArg.tInfo

    if printBase:
        if tInfo.status: 
            gm.tabprint("%s - %s - %d Models" % ( tInfo.status, tInfo.get('target_id'), len( tInfo.tDict['zoo_merger_models'] ) ) )

    # Check if valid directory
    if tInfo.status == False:
        gm.eprint("WARNING: SIMR.target_initialize:  Target Info status bad")
        return None
            
    if cmdArg.get('printParam', False):
        tInfo.printParams()
    
    # Check if target needs to create a new image.

    return tInfo

# End target_initialize

def prep_score_parameters( cmdArg, tInfo ):
    
    ###################################
    ###  Validate Score Parameters  ###
    ###################################
    
    printBase = cmdArg.printBase
    printAll  = cmdArg.printAll
    
    # Check if valid score parameters.
    scoreParams = cmdArg.get('scoreParams')
    scoreParamName = cmdArg.get('scoreParamName',None)
    scoreParamLoc = gm.validPath( cmdArg.get('scoreParamLoc',None) )
    
    if printAll:
        print("SIMR: prep_score_parameters: ")
        gm.tabprint('ParamType: %s'%type(scoreParams))
        gm.tabprint('scoreParamName: %s'%scoreParamName)
        gm.tabprint('scoreParamLoc : %s'%scoreParamLoc)
        
    
    # If params there, move on
    if scoreParams != None:
        pass
    
    # If invalid, complain
    elif scoreParams == None and scoreParamLoc == None and scoreParamName == None:
        if printBase:
            gm.eprint("WARNING: prep_score_parameters: params not valid")
            gm.etabprint('ParamType: %s'%type(scoreParams))
            gm.etabprint('scoreParamName: %s'%scoreParamName)
            gm.etabprint('scoreParamLoc : %s'%scoreParamLoc)
        return None

    # If given a param name, assume target knows where it is.
    elif scoreParams == None and scoreParamName != None:
        
        # Check if in target's score parameters
        scoreParams = tInfo.getScoreParam( scoreParamName )
        gm.tabprint("From target_info: ", type(scoreParams)) if printAll else None
        
        # Else look for file by that name
        if scoreParams == None:
            scoreParams = tInfo.readScoreParam(scoreParamName)
            gm.tabprint("From target file: ", type(scoreParams)) if printAll else None
    
    # If given param location, directly read file
    elif scoreParamLoc != None:
        scoreParams = gm.readJson(scoreParamLoc)

    # Check for final parameter file is valid
    if scoreParams == None:
        if printBase:
            gm.eprint("WARNING: SIMR.prep_score_parameters:")
            gm.etabprint("Failed to load parameter class")
        return None
    
    cmdArg.setArg('scoreParams', scoreParams)
    
    # If needing new target image
    if cmdArg.get( 'newTargetImage', False ):
        if printBase:
            print("SIMR.prep_score_parameters: Creating new target image")
        
        # Loop through all params for creation
        for key in scoreParams:
            new_param = IMG.adjustTargetImage( tInfo = tInfo, new_param = scoreParams[key], \
                                 overWrite = cmdArg.get('overWrite'), printAll = printAll )
            if new_param != None:
                scoreParams[key] = new_param
                
            else:
                gm.eprint("WARNING: SIMR.prep_score_parameters:")
                gm.etabprint("Bad new parameters")
                
    # Print param stuff
    cmdArg.setArg('scoreParams', scoreParams)
    tInfo.addScoreParameters( scoreParams )

    if printAll:
        print("SIMR: prep_score_parameters: Finished Prepping ")
        gm.tabprint('scoreParams: %s'%type(scoreParams))
        gm.tabprint('scoreParamName: %s'%scoreParamName)
        gm.tabprint('scoreParamLoc : %s'%scoreParamLoc)
            
    return cmdArg.get('scoreParams',None)
# End prep_score_parameters    


def target_prep_cmd_params( tInfo, tArg ):

    printBase = tArg.printBase
    printAll = tArg.printAll
    
    if printBase:
        print("SIMR: target_prep_cmd_params: %s" % tInfo.get('target_id') )

    if tInfo.printAll:
        gm.tabprint( '%d in target_prep_cmd_params: %s: '% (mpi_rank, tInfo.get('target_id') ) ) 
    
    # Copy cmd Arg to send to runs
    runArgs = deepcopy( tArg )
    
    # Give model runs knowledge of target info
    runArgs.setArg('tInfo', tInfo)
    
    # Change printing variables
    if tArg.get('printAllRun',False):
        runArgs.setArg('printBase',True)
        runArgs.setArg('printAll',True)  
        
    elif tArg.get('printBaseRun',False):
        runArgs.setArg('printBase',True)
        runArgs.setArg('printAll',False)     
    
    # Default is to not have all 1000's of runs print something
    else:
        runArgs.setArg('printBase',False)
        runArgs.setArg('printAll',False)
    
    if printAll:
        print('SIMR.target_prep_cmd_params: Printing new run argments\n')
        runArgs.printArg()
        print('')
    
    return runArgs

def target_new_orbits( tInfo, cmdArg ):
    
    printBase = cmdArg.printBase
    printAll = cmdArg.printAll
    
    if printAll:
        print( "SIMR: target_new_orbits:" ) 
        
    # If not in MPI Env
    if mpi_size == 1:
        
        runArgs = target_prep_cmd_params( tInfo, cmdArg )
        
    
    # In MPI_Env
    else:
                
        # Have rank 0 prep args and broadcast
        if mpi_rank == 0:
            runArgs = target_prep_cmd_params( tInfo, cmdArg )
            
        else:
            runArgs = None
            
        runArgs = mpi_comm.bcast( runArgs, root=0 )
        mpi_comm.Barrier()
    
    if type(runArgs) == type(None):
        print("WARNING: SIMR.target_new_scores: Invalid run arguments")
        return None
    
    
    # Rank 0 has argList and will distribute
    argList = None    
    scatter_lists = []
    
    if mpi_rank == 0:
        
        
        # Check if grabbing subset of runs.  If not, presume all
        if cmdArg.get('startRun',None) != None \
        and cmdArg.get('nRun',None) != None \
        and cmdArg.get('endRun',None) != None:
         
            runDicts = tInfo.iter_run_dicts()
        
        else:            
            runDicts = tInfo.iter_run_dicts( \
               startRun = cmdArg.get('startRun',0), \
               endRun   = cmdArg.get('endRun',-1), \
               stepRun  = cmdArg.get('stepRun',1),)
        
        argList = []
        
        for i,rKey in enumerate(runDicts):

            rDir = tInfo.getRunDir(rID=rKey)
            if rDir == None:
                if printBase: 
                    gm.eprint("WARNING: SIMR.target_new_orbits: Run invalid: %s" % rKey)
                    
            argList.append( dict( cmdArg = runArgs, rDir=rDir, ) )

        # Print how many scores expecting to be completed.
        if tInfo.printBase:      
            gm.tabprint("Runs needing orbits: %d"%len(argList))

        # Divide up the big list into many lists for distributing
        if len( argList ) > 0:
            scatter_lists = [ argList[i::mpi_size] for i in range(mpi_size) ]

        # If nothing to do, create a list of empty lists so others know
        else: 
            for i in range(mpi_size):
                scatter_lists.append([])

        if tInfo.printAll:
            print("SIMR: target_new_scores: MPI Scatter argList")
            gm.tabprint("Rank 0 argList: %d"%len(argList))
            gm.tabprint("Rank 0 scatter_lists: %d"%len(scatter_lists))
            for i,lst in enumerate(scatter_lists):
                gm.tabprint("Rank 0 list %d: %d"%(i,len(lst)))

    # Scatter argument lists to everyone
    argList = mpi_comm.scatter(scatter_lists,root=0)
    if tInfo.printAll:
        gm.tabprint("Rank %d received: %d"%(mpi_rank,len(argList)))
        
        
    ############################
    ###  Execute Model Runs  ###
    ############################

    # Everyone go through their list and execute runs    
    for i,args in enumerate(argList):
        run_new_score( **args )

        if mpi_rank == 0:
            gm.tabprint("Rank 0: Progress: %d / %d " % (i+1,len(argList)), end='\r')

    if mpi_rank == 0: print('')

    # Everyone wait for everyone else to finish
    mpi_comm.Barrier()

    # Have rank 0 collect files and update scores
    if mpi_rank == 0:

        # Check if target needs to create feature values
        if cmdArg.get('newFeats',False):
            fe.wndchrm_target_all( cmdArg, tInfo )
            fe.reorganize_wndchrm_target_data( cmdArg, tInfo )
        
        # Normalize feature values after created for runs
        if cmdArg.get('normFeats',False):
            fe.target_collect_wndchrm_all_raw( cmdArg, tInfo = tInfo )
            fe.target_wndchrm_create_norm_scaler( cmdArg, tInfo, )
            
        # Create new neural network model if called for
        if cmdArg.get('newNN',False):
            nn.target_new_neural_network( cmdArg, tInfo )

        tInfo.gatherRunInfoFiles()
        tInfo.updateScores()

    
# End target_new_orbit
    
        

def target_new_scores( tInfo, cmdArg ):
    
    printBase = cmdArg.printBase
    printAll = cmdArg.printAll
    
    if printAll:
        print( "SIMR: target_new_scores:" )
    
    
    ###############################
    ###  Prepare Run Arguemnts  ###
    ###############################
    
    
    runArgs = None
    
    # If not in MPI Env
    if mpi_size == 1:
        
        prep_score_parameters( cmdArg, tInfo )
        runArgs = target_prep_cmd_params( tInfo, cmdArg )
    
    # In MPI_Env
    else:
        
        # Have rank 0 prep args and broadcast
        if mpi_rank == 0:
            
            prep_score_parameters( cmdArg, tInfo )
            runArgs = target_prep_cmd_params( tInfo, cmdArg )
            mpi_comm.bcast( runArgs, root=0 )

        # If in mpi and expecting info.
        else:
            runArgs = mpi_comm.bcast( runArgs, root=0 )

        mpi_comm.Barrier()
    
    if type(runArgs) == type(None):
        print("WARNING: SIMR.target_new_scores: Invalid run arguments")
        return None


    # Rank 0 has argList and will distribute
    argList = None    
    scatter_lists = []
    
    if mpi_rank == 0:
        
        # Find out which runs need new scores
        tInfo.gatherRunInfoFiles()
        
        # Check if grabbing subset of runs.  If not, presume all
        if cmdArg.get('startRun',None) != None \
            and cmdArg.get('nRun',None) != None \
            and cmdArg.get('endRun',None) != None:
         
            runDicts = tInfo.iter_run_dicts()
        
        else:            
            runDicts = tInfo.iter_run_dicts( \
                                           startRun = cmdArg.get('startRun',0), \
                                           endRun   = cmdArg.get('endRun',-1), \
                                           stepRun  = cmdArg.get('stepRun',1),)
        
        argList = []
        for i,rKey in enumerate(runDicts):
            
            rScore = tInfo.get('zoo_merger_models')[rKey]['machine_scores']

            # Loop through wanted scores
            scoreGood = True
            for sKey in runArgs.get('scoreParams',{}):
                if rScore.get(sKey,None) == None:
                    scoreGood = False
                    break

            if not scoreGood or cmdArg.get('overWrite',False):
                rDir = tInfo.getRunDir(rID=rKey)
                if rDir == None:
                    if printBase: print("WARNING: Run invalid: %s" % rKey)
                argList.append( dict( cmdArg = runArgs, rDir=rDir, ) )

        # Print how many scores expecting to be completed.
        if tInfo.printBase:
            if len(argList) == 0:
                gm.tabprint("Scores already exist!")        
            else: 
                gm.tabprint("Runs needing scores: %d"%len(argList))

        # Divide up the big list into many lists for distributing
        if len( argList ) > 0:
            scatter_lists = [ argList[i::mpi_size] for i in range(mpi_size) ]

        # If nothing to do, create a list of empty lists so others know
        else: 
            for i in range(mpi_size):
                scatter_lists.append([])

        if tInfo.printAll:
            print("SIMR: target_new_scores: MPI Scatter argList")
            gm.tabprint("Rank 0 argList: %d"%len(argList))
            gm.tabprint("Rank 0 scatter_lists: %d"%len(scatter_lists))
            for i,lst in enumerate(scatter_lists):
                gm.tabprint("Rank 0 list %d: %d"%(i,len(lst)))

    # Scatter argument lists to everyone
    argList = mpi_comm.scatter(scatter_lists,root=0)
    if tInfo.printAll:
        gm.tabprint("Rank %d received: %d"%(mpi_rank,len(argList)))
        
        
    ############################
    ###  Execute Model Runs  ###
    ############################

    # Everyone go through their list and execute runs    
    for i,args in enumerate(argList):
        run_new_score( **args )

        if mpi_rank == 0:
            gm.tabprint("Rank 0: Progress: %d / %d " % (i+1,len(argList)), end='\r')

    if mpi_rank == 0: print('')

    # Everyone wait for everyone else to finish
    mpi_comm.Barrier()

    # Have rank 0 collect files and update scores
    if mpi_rank == 0:

        # Check if target needs to create feature values
        if cmdArg.get('newFeats',False):
            fe.wndchrm_target_all( cmdArg, tInfo )
            fe.reorganize_wndchrm_target_data( cmdArg, tInfo )
        
        # Normalize feature values after created for runs
        if cmdArg.get('normFeats',False):
            fe.target_collect_wndchrm_all_raw( cmdArg, tInfo = tInfo )
            fe.target_wndchrm_create_norm_scaler( cmdArg, tInfo, )
            
        # Create new neural network model if called for
        if cmdArg.get('newNN',False):
            nn.target_new_neural_network( cmdArg, tInfo )

        tInfo.gatherRunInfoFiles()
        tInfo.updateScores()

# End processing target dir for new scores


def run_new_score( cmdArg = None, rInfo = None, rDir = None ):

    # Initialize variables
    if cmdArg == None:
        print("SIMR: WARNING: No arg. Exiting")

    if rDir == None:
        rDir = cmdArg.runDir
        
    printAll = cmdArg.printAll
    printBase = cmdArg.printBase    
    
    if printBase:
        print("SIMR.run_new_score: ")
    
    if printAll:
        print("\t - rDir:", rDir)
        print("\t - rInfo:", type(rInfo) )

    # Initialize info file
    if rInfo == None:
        
        if cmdArg.get('rInfo') != None:
            rInfo = cmdArg.rInfo
            
        elif rDir != None:
            rInfo = im.run_info_class( runDir=rDir, rArg=cmdArg )

    if printAll:
        print('SIMR.run_new_score: ')
        print('\t - rInfo: ', (rInfo) )

    if rInfo.status == False:
        if printBase: print("SIMR.run_new_score: WARNING: runInfo bad")
        return None
    
    if printBase:
        gm.tabprint('runID: %s' % rInfo.get('run_id'))
    
    # Check if asking for new orbit
    if cmdArg.get('newOrbit'):
        SIM.orbit_simulation( rInfo = rInfo, cmdArg=tmpArg )
    
    # Check if score parameters were given
    if cmdArg.get('scoreParamLoc') != None and cmdArg.get('scoreParams') == None:
        sParams = gm.readJson( cmdArg.scoreParamLoc )
        if sParams == None:
            if cmdArg.printBase: print("ERROR: SIMR: run_new_score: Error reading param file: %s"%cmdArg.scoreParamLoc )
        else:
            cmdArg.scoreParams = sParams
    
    # If invalid, complain
    if cmdArg.get('scoreParams',None) == None:
        if printBase:
            print("SIMR: WARNING: run_new_score: params not valid")
        return
    
    # Check for new scores to create
    overWrite =  cmdArg.get('overWrite', False) or cmdArg.get('overwrite', False)
    scoreParams = cmdArg.get('scoreParams',None)
    newParams = {}
    
    if overWrite:
        if rInfo.printAll: im.tabprint("Overwriting old scores")
        newParams = deepcopy( scoreParams )
        
    # Loop through score parameters and grab unique simulation scenarios. 
    else:
        for key in scoreParams:
            
            # Check for valid param
            if type( scoreParams[key] ) == type( None ) or type( scoreParams[key] ) != type( {'dict':'dict'} ) :
                gm.etabprint("WARNING: SIMR.run_new_score: %s" % rInfo.get('run_id') )
                gm.etabprint("Invalid Score Param: %s" % key )
                gm.etabprint(scoreParams[key]) 
                continue
                
            score = rInfo.getScore( scoreParams[key]['name'] )
            if score == None:
                newParams[key] = scoreParams[key]
        # End simKey in simParams
    
    # No new scores
    if len( newParams ) == 0:
        
        if rInfo.printBase:
            im.tabprint("No new scores")
        return
    
    else:  
        if rInfo.printAll:  gm.tabprint("Creating new scores")
        
    tmpArg = deepcopy( cmdArg )
    tmpArg.scoreParams = newParams

    # Check if new files should be created/altered    
    
    newAll = cmdArg.get('newAll',False)
    
    # Asking for new simulation data?
    if cmdArg.get('newSim') or newAll:
        SIM.main_sim_run( rInfo = rInfo, cmdArg=tmpArg )

    # Asking for new image creation?
    if cmdArg.get('newImage') or newAll:
        IMG.main_img_run( rInfo = rInfo, arg=tmpArg )
        
    # Asking for new feature extraction from images? 
    if cmdArg.get('newFeats') or newAll:
        fe.main_fe_run( rInfo = rInfo, arg=tmpArg )

    # Asking for new score based on sim, image, and/or feature extractions?
    if cmdArg.get('newScore') or newAll:
        ms.MS_Run( printBase = printBase, printAll = printAll, \
                rInfo = rInfo, params = tmpArg.get('scoreParams'), \
                arg = tmpArg )
    
    # Clean up temporary files from run directory.
    rInfo.delTmp()
    
    return rInfo
    

# end processing run



# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    
    main( arg )
