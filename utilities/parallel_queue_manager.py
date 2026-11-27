#!/usr/bin/env python
# coding: utf-8

"""
File: mpi_queue_manager.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2024-Nov-11

Description: This code abstacts my parallel queue processing. 

References:  
    Code is based on example: https://github.com/luca-s/mpi-master-slave/blob/master/examples/example1.py
    Sections of this code were enhanced with the assistance of ChatGPT made by OpenAI.
"""

# ================================ IMPORTS ================================ #
# Standard library imports
import logging
import os
import sys
import time

import numpy as np
import dill
import concurrent.futures
from mpi4py import MPI 
from mpi_master_slave import Master, Slave, WorkQueue

# Add main project directory to import project modules
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
sys.path.append(PROJECT_DIRECTORY)

# import project modules
import utilities.general_utility as gu

# ================================ GLOBAL VARIABLES ================================ #
logger = logging.getLogger()

# Define MPI tags
TAG_TASK = 1
TAG_ERROR = 2
TAG_RESULT = 3
TAG_TERMINATE = 4

# Define Local concurrent.future variables
_local_eval_function = None

def _local_worker_initializer(serialized_func):
    global _local_eval_function
    _local_eval_function = dill.loads(serialized_func)

# ================================ CLASSES ================================ 

class Queue_Manager():

    def __init__(self, in_function, batch_size=1, hard_mode = None,
                 function_logger=None, function_log_level=logging.WARNING, queue_log_level = logging.INFO
                ):

        # Save inputs for later
        self.eval_function = in_function
        self.batch_size = batch_size

        # Initialize a custom queue logger with class name in formatting
        self.logger = gu.configure_logger( logger_name='Queue_Manager',
                                        log_level=queue_log_level,
                                            )

        # if None, assume Root logger, otherwise use provided Logger 
        self.func_logger = function_logger if function_logger is not None else logging.getLogger('')
        self.func_log_level = function_log_level

        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.master = None

        # Set local machine variables
        self.cpu_count = os.cpu_count()
        self.serialized_func = dill.dumps(self.eval_function)
        self.executor = None
        
        # Determine mode:
        # If hard coded, use that
        if hard_mode == 'mpi':
            self._set_mpi_mode()
        elif hard_mode == 'local':
            self._set_local_mode()
        elif hard_mode == 'solo':
            self._set_solo_mode()

        # MPI mode if size > 1
        # Local futures mode if size < 2 and cpu_count > 1
        # Single-core mode if size < 2 and cpu_count == 1
        elif hard_mode is None:
            if self.size > 1:
                self._set_mpi_mode()
            elif self.size < 2 and self.cpu_count > 1:
                self._set_local_mode()
            else:
                self._set_solo_mode()
        else:
            self.logger.error("Invalid Parallel Processing Mode: %s" % hard_mode)
            self.logger.error("Valid Modes: 'mpi', 'local', 'solo'")
            raise ValueError("Invalid Parallel Processing Mode %s" % hard_mode)
    # end __init__

    def __del__(self):

        # Terminate workers is master is not None
        if self.master is not None:
            self.terminate_workers()

        # Shutdown Executor if not None
        if self.executor is not None:
            self.executor.shutdown(wait=True)

        self.logger.info(f"Queue Manager Deleted")
    # end __del__

    def terminate_workers(self):
        if self.rank == 0 and self.size > 1:
            self.master.terminate_slaves()
            self.master = None
            self.queue = None
    # end terminate_workers

    def evaluate_tasks(self, task_list):
        """
        Determine which run method to call based on the mode.
        """

        # validate task_list
        if isinstance(task_list, list):
            n = len(task_list)
        elif isinstance(task_list, np.ndarray):
            n = task_list.shape[0]
        else:
            self.logger.error("Invalid task_list type: %s" % type(task_list))
            self.logger.error("Valid types: list, numpy.ndarray")
            raise ValueError("Invalid task_list type: %s" % type(task_list))
        
        # Verify there are tasks to run
        if n == 0:
            self.logger.error("No tasks to evaluate")
            return []

        # Change function logger level
        prev_level = self.func_logger.level
        self.func_logger.setLevel( self.func_log_level )

        # Run tasks based on mode   
        if self.mode == 'mpi':
            results = self._run_mpi_queue(task_list)
        elif self.mode == 'local':
            results =  self._run_local_processes(task_list)
        elif self.mode == 'solo':  # single_core
            results =  self._run_single_core(task_list)
        else:
            self.func_logger.setLevel( prev_level )
            self.logger.error("Invalid Parallel Processing Mode: %s" % self.mode)
            self.logger.error("Valid Modes: 'mpi', 'local', 'solo'")
            raise ValueError("Invalid Parallel Processing Mode %s" % self.mode)
    
        # Return to previous level.
        self.func_logger.setLevel( prev_level )

        # Print number of errors returned
        error_list = [(i,e) for i,e in enumerate(results) if isinstance(e, Exception)]
        if len(error_list) > 0:
            self.logger.error(f"Errors Returned: {len(error_list)} of {len(results)} results")
            c = 0
            for i,e in error_list:
                self.logger.error(f"Error {i}: {e}")
                c += 1
                if c > 2:
                    self.logger.error("...")
                    break
                

        # return results
        return results
    
    # end evaluate_tasks

    def _set_solo_mode(self,):
        self.mode = 'solo'
        self.logger.info("Solo Loop Mode.")
    # end _set_solo_mode

    def _set_mpi_mode(self,):

        self.mode = 'mpi'
        if self.master is None:
            self.master = Master(range(1, self.size))
            self.queue = WorkQueue(self.master)
        self.wait_time = 0.01
        self.logger.info(f"MPI Manager initialized with {self.size} Workers in MPI mode.")
    # end _set_mpi_mode

    def _set_local_mode(self,):
        self.mode = 'local'

        if self.executor is None:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.cpu_count,
                initializer=_local_worker_initializer,
                initargs=(self.serialized_func,)
            )

        self.logger.info(f"Local Manager initialized with {self.cpu_count} workers using local multiprocessing (futures).")
    # end _set_local_mode
    
    def set_parallel_mode(self, mode):
        if mode == 'mpi':
            self._set_mpi_mode()
        elif mode == 'local':
            self._set_local_mode()
        elif mode == 'solo':
            self._set_solo_mode()
        else:
            self.logger.error("Invalid Parallel Processing Mode: %s" % mode)
            self.logger.error("Valid Modes: 'mpi', 'local', 'solo'")
            raise ValueError("Invalid Parallel Processing Mode %s" % mode)

    def gen_simple_progress_string( self, n_tasks, n_completed, length=20, char='=' ):
        """
        Generate a simple progress string
        """
        n = int( n_completed / n_tasks * length )
        return f"[{char*n}{' '*(length-n)}] {n_completed}/{n_tasks}"
    # end gen_simple_progress_string


    def _run_single_core(self, task_list):
        """
        If running on a single core and size < 1 or size=1 with no extra cores,
        simply loop through the evaluation function.
        """
        n = len(task_list)
        compiled_results = {}
        curr_time = time.time()

        # Initial progress print
        prog_str = self.gen_simple_progress_string(n, 0)
        self.logger.info("Solo Manager: Progress: %s" % (prog_str))

        # Loop through tasks
        for i, t in enumerate(task_list):
            try:
                compiled_results[i] = self.eval_function(t)
            except Exception as e:
                self.logger.warning(f"Queue Task {i} Failed: '{e}'")
                compiled_results[i] = e

            if time.time() - curr_time > 1:
                prog_str = self.gen_simple_progress_string(n, len(compiled_results))
                self.logger.info("Solo Manager: Progress: %s" % (prog_str))
                curr_time = time.time()

        prog_str = self.gen_simple_progress_string(n, len(compiled_results))
        self.logger.info("Solo Manager: Progress: %s - Queue Complete" % (prog_str))
        return [compiled_results[i] for i in range(n)]
    # end _run_single_core

    def _run_mpi_queue(self, task_list):
        n = len(task_list)
        curr_time = time.time()
        compiled_results = {}

        # Add tasks to MPI queue
        for i, task in enumerate(task_list):
            self.queue.add_work(data=(i, task))
        
        # Do initial progress print
        prog_str = self.gen_simple_progress_string(n, 0)
        self.logger.info("MPI Manager: Progress: %s" % (prog_str))

        # Keep 'poking' workers while there is work to do
        while not self.queue.done():
            self.queue.do_work()

            completed_results = self.queue.get_completed_work()
            for result in completed_results:
                worker_done, worker_results = result
                if worker_done:
                    idx, return_value = worker_results
                    self.logger.debug(f'Manager: Received results {idx}:{return_value}')
                    compiled_results[idx] = return_value

            if time.time() - curr_time > 1:
                prog_str = self.gen_simple_progress_string(n, len(compiled_results))
                self.logger.info("MPI Manager: Progress: %s" % (prog_str))
                curr_time = time.time()

            time.sleep(self.wait_time)

        # Queue is empty
        prog_str = self.gen_simple_progress_string(n, len(compiled_results))
        self.logger.info("MPI Manager: Progress: %s - Queue Complete" % (prog_str))
        return [compiled_results[i] for i in range(n)]
    # end _run_mpi_queue

    @staticmethod
    def _run_local_task(task):
        global _local_eval_function
        try:
            return _local_eval_function(task)
        except Exception as e:
            return e
    # end _run_local_task

    def _run_local_processes(self, task_list):
        n = len(task_list)
        curr_time = time.time()
        compiled_results = [None] * n  # Pre-allocate a list for ordered results

        # Submit all tasks and map futures to their indices
        future_to_idx = {
            self.executor.submit(Queue_Manager._run_local_task, task): i for i, task in enumerate(task_list)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                compiled_results[i] = future.result()
            except Exception as e:
                self.logger.warning(f"Queue Task {i} failed: '{e}'")
                compiled_results[i] = e

                # Periodic progress update
            if time.time() - curr_time > 1:
                prog_str = self.gen_simple_progress_string(n, sum(r is not None for r in compiled_results))
                self.logger.info(f"Local Manager: Progress: {prog_str}")
                curr_time = time.time()

        # Final progress update
        prog_str = self.gen_simple_progress_string(n, sum(r is not None for r in compiled_results))
        self.logger.info(f"Local Manager: Progress: {prog_str} - Queue Complete")

        return compiled_results  # Guaranteed to be in the same order as task_list
    # end _run_local_processes
    
# end Queue_Manager

class Queue_Worker(Slave):
    """
    Worker extends Slave class, overrides the 'do_work' method
    and calls 'Slave.run'. The Master will do the rest
    """

    def __init__(self, in_function, worker_log_level=logging.INFO ):
        super(Queue_Worker, self).__init__()

        # Get MPI rank
        self.rank = MPI.COMM_WORLD.Get_rank()

        # Create logger
        self.logger = gu.configure_logger( logger_name=f'Queue_Worker_{self.rank}',
                                            log_level=worker_log_level,
                                            )

        # Save dictionary of possible functions to run
        self.eval_function = in_function
        self.logger.info(f"Queue Worker {self.rank}: Initialized")

        # Start worker
        self.run()

    def __del__(self):
        self.logger.info(f"Queue Worker {self.rank}: Deleted")
    
    def do_work(self, in_message):

        task_idx, task_input = in_message

        self.logger.debug( f"Worker {self.rank}: Received task {task_idx}" )

        try:
            # Run function
            task_output = self.eval_function(task_input)

        except Exception as e:
            self.logger.warning( f"Queue Task {task_idx} failed: \n{e}" )
            task_output = e

        self.logger.debug( f"Worker completed task: {task_idx}" )
        return (True, (task_idx, task_output))
# end Queue_Worker

# ================================ FUNCTIONS ================================ #
# setup two functions functions to evaluate
def test_func_1(x, a=1, b=2):
    time.sleep( 0.25 )
    return np.sum( a * np.power(x, b) )

# Computationally expensive function
def test_func_2(x, a=1, b=2):
    # randomly wait between 0 and 10 seconds
    time.sleep( np.random.random() * 5 )
    return np.sum( a * np.power(x, b) ) * np.random.random()

# Wrapper around both functions for demonstration
def func_wrapper( w_input ):
    f_id, f_input = w_input
    if f_id == 1:
        return test_func_1(f_input)
    elif f_id == 2:
        return test_func_2(f_input)
# end func_wrapper

# ================================ TESTING ================================ #
def test_queue( ):

    # Example function calls
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Have only rank 0 print to avoid duplicate prints
    if rank == 0:
        time.sleep(1)
        print("\nTESTING QUEUE MANAGER\n")

    # Setup Workers, and have them on standby
    if rank != 0:
        worker = Queue_Worker( func_wrapper )
        del worker
    
    # Setup manager and run tests
    if rank == 0:
        try:
            # Solo mode
            time.sleep(1)  # Let workers initialize and print their messages
            print("\nTEST 1: SOLO MODE\n")
            start_time = time.time()
            manager = Queue_Manager( func_wrapper, hard_mode='solo' )
            n = 4
            test_list = [ (2, np.random.random(10)) for i in range(n) ]
            results = manager.evaluate_tasks( test_list )
            time_taken = time.time() - start_time
            time.sleep(1)

            print(f"\nTest 1 Results: {len(results)/time_taken:.2f} tasks per second")
            gu.tabprint(f"Results: {len(results)}")
            gu.tabprint(f"Time Taken: {time_taken:.2f} seconds")
            gu.tabprint(f"Num Cores: 1")
            time.sleep(1)
            
            # Local mode
            print("\n# ================================ #\n")
            print("\nTEST 2: LOCAL MACHINE MODE\n")
            manager.set_parallel_mode('local')
            n = 120
            test_list = [ (2, np.random.random(10)) for i in range(n) ]
            start_time = time.time()
            results = manager.evaluate_tasks( test_list )
            time_taken = time.time() - start_time
            time.sleep(1)

            print(f"\nTest 2 Results: {len(results)/time_taken:.2f} tasks per second")
            gu.tabprint(f"Results: {len(results)}")
            gu.tabprint(f"Time Taken: {time_taken:.2f} seconds")
            gu.tabprint(f"Num Cores: {manager.cpu_count}")
            time.sleep(1)

            # MPI mode
            print("\n# ================================ #\n")
            print("\nTEST 3: MPI MODE\n")
            manager.set_parallel_mode('mpi')
            n = 12
            test_list = [ (2, np.random.random(10)) for i in range(n) ]
            start_time = time.time()
            results = manager.evaluate_tasks( test_list )
            time_taken = time.time() - start_time
            time.sleep(1)

            print(f"\nTest 3 Results: {len(results)/time_taken:.2f} tasks per second")
            gu.tabprint(f"Results: {len(results)}")
            gu.tabprint(f"Time Taken: {time_taken:.2f} seconds")
            gu.tabprint(f"Num Cores: {manager.size}")
            print('')

            # Test termination
            manager.terminate_workers()
            time.sleep(1)       
            del manager
        except Exception as e:
            logger.critical("Error in test_queue: %s" % e)

# ================================ MAIN ================================ #
# If script is called main
if __name__ == '__main__':

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("\PARALLEL PROCESSING QUEUE MANAGER\n")
        gu.tabprint("This module is intended to be imported and used in other scripts.")
        gu.tabprint("Calling as main will run tests and examples.")
        print("\nInitializing Arguments, and Logger\n")

    try:
        args, logger = gu.initialize_environment()
        # gu.change_logging_level('DEBUG')

    except:
        print("Failed to Initalize Arguments and Logger")
        sys.exit(1)

    # Run tests
    test_queue()

    comm.Barrier()
    if rank == 0:
        print("\nEnd of Script\n")
