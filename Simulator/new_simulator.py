#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: new_simulator.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden 
Created: 2019-May-10

Description: This script is designed to handle calling the SPAM fortran simulator

References:  Sections of this code were written with the assistance 
    of ChatGPT made by OpenAI.

"""
# ================================ IMPORTS ================================ #

# Standard library imports
import hashlib, logging, os, sys
import numpy as np


# Add main project directory based on current script location.
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))  # Grab file location
PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY) # project directory one layer up
sys.path.append(PROJECT_DIRECTORY)

# import project modules
import utilities.general_utility as gu
import utilities.model_manager as mm
from Simulator.custom_runs import custom_runs_module as spam_module
spam_module.simr_init()

# ================================= GLOBALS ================================= #
# Global logger from main program
LOGGER = gu.configure_logger(logger_name='')

# An example collision model
test_param = [-9.93853,-4.5805,3.27377,-0.50008,-2.45565,-1.07799,23.33004,24.69427,3.49825,5.32056,309.6923,36.8125,41.78471,51.42857,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.94594,0,0,0.18901,-22.47999,9.02372,0.0,1.0,0.0,0.0,0.0,0.0][0:22]

#   Standardized parameter array for SPAM model:
spam_param_description = '''
    [0]: X-coordinate of the secondary galaxy's position
    [1]: Y-coordinate of the secondary galaxy's position
    [2]: Z-coordinate of the secondary galaxy's position
    [3]: X-component of the secondary galaxy's velocity
    [4]: Y-component of the secondary galaxy's velocity
    [5]: Z-component of the secondary galaxy's velocity
    [6]: Mass of the primary galaxy
    [7]: Mass of the secondary galaxy
    [8]: Outer radius of the primary galaxy
    [9]: Outer radius of the secondary galaxy
    [10]: Azimuthal angle for the primary galaxy
    [11]: Azimuthal angle for the secondary galaxy
    [12]: Inclination angle for the primary galaxy
    [13]: Inclination angle for the secondary galaxy
    [14]: Softening length for the primary galaxy
    [15]: Softening length for the secondary galaxy
    [16]: Scaling factor for bulge in the primary galaxy
    [17]: Scaling factor for disk in the primary galaxy
    [18]: Scaling factor for halo in the primary galaxy
    [19]: Scaling factor for bulge in the secondary galaxy
    [20]: Scaling factor for disk in the secondary galaxy
    [21]: Scaling factor for halo in the secondary galaxy
'''


# ================================= CORE FUNCTIONS ================================= #
def set_spam_parameters( lnl = 0.1, r_scale = 10.0, m_halo = 5.8, m_bulge = 0.3333, h_bulge = 2.0, h_disk = 1.0 ):
    """
    This function defines the base mass distribution of the galaxy.   Custom collisions scale this distribution.
    The default values are derived from empirical data of the Milky Way and M31.  
    See https://ui.adsabs.harvard.edu/abs/1993ApJS...86..389H/abstract
    
    Parameters:
        lnl (float): Variable to adjust the strength of the dynamic friction
        r_scale (float): Radial scaling factor for mass distribution of galaxy
        m_halo (float): Mass of the halo
        m_bulge (float): Mass of the bulge
        h_bulge (float): Exp. decay rate of bulge density profile
        h_disk (float): Exp. decay rate of disk density profile

    Returns:
        None
    """

    # Call the Fortran function
    LOGGER.info("Setting SPAM Parameters: ")
    LOGGER.debug("lnl: %.2e, r_scale: %.2f, m_halo: %.2f, m_bulge: %.2f, h_bulge: %.2f, h_disk: %.2f" % (lnl, r_scale, m_halo, m_bulge, h_bulge, h_disk))

    try:
        spam_module.simr_init_distribution(lnl, r_scale, m_halo, h_disk, h_bulge, m_bulge)

    except:
        LOGGER.error("Failed to call 'spam_module.simr_init_distribution'")
        LOGGER.error(f"lnl: {lnl}, r_scale: {r_scale}, m_halo: {m_halo}, m_bulge: {m_bulge}, h_bulge: {h_bulge}, h_disk: {h_disk}")
        LOGGER.error(f"Input Types: {type(lnl)}, {type(r_scale)}, {type(m_halo)}, {type(m_bulge)}, {type(h_bulge)}, {type(h_disk)}")
        LOGGER.error(f"Exception: \n",exc_info=True)
        raise ValueError("Failed to call 'spam_module.simr_init_distribution'") from e

    return

# set_spam_parameters() # Run once to set the default values

def set_spam_parameters_from_dict( spam_params ):
    
    # Setup the simulation paramaters if given
    if spam_params is not None:
        # Create new dict with only valid keys
        valid_keys = ['lnl', 'r_scale', 'm_halo', 'm_bulge', 'h_bulge', 'h_disk']
        valid_setup_params = {k:spam_params[k] for k in valid_keys if k in spam_params}
        set_spam_parameters( **valid_setup_params ) # Missing keys will use default value from function
    

def get_spam_parameters():
    """
    This function returns the current SPAM parameters

    Returns:
        spam_params (dict): Dictionary of the current SPAM parameters
    """
    LOGGER.info("Querying SPAM Parameters")

    keys = ['lnl', 'r_scale', 'm_halo', 'm_bulge', 'h_bulge', 'h_disk']
    spam_values = spam_module.get_spam_parameters()
    spam_parameters = {}

    # Create dictionary of parameters
    for i, key in enumerate(keys):
        LOGGER.debug(f"{key}: {spam_values[i]}")
        spam_parameters[key] = spam_values[i]
    
    return spam_parameters

# print( get_spam_parameters() )

def initialize_setup( collision_param, spam_setup_params = None ):

    LOGGER.debug("Initializing Collision Parameters and Simulation Variables")

    # validate the collision_param is a 1D list or np.array of size 22
    if not isinstance(collision_param, (list, np.ndarray)):
        LOGGER.error("Collision parameters must be a list or numpy array")
        LOGGER.error("Collision Param Type: %s" % type(collision_param))
        raise ValueError("Collision parameters must be a list or numpy array")
        
    if len(collision_param) != 22:
        LOGGER.error("Collision parameters must be a list or numpy array of size 22")
        LOGGER.error("Collision Param Length: %d" % len(collision_param))
        raise ValueError("Collision parameters must be a list or numpy array of size 22")
    
    # ensure array is in the correct format for fortran
    fortran_ar = np.asfortranarray(np.array(collision_param).astype(np.float64))
    
    # Setup the simulation paramaters if given
    if spam_setup_params is not None:
        # Create new dict with only valid keys
        spam_setup_params = dict(spam_setup_params) # Ensure it's a dict
        valid_keys = ['lnl', 'r_scale', 'm_halo', 'm_bulge', 'h_bulge', 'h_disk']
        valid_setup_params = {k:float(spam_setup_params[k]) for k in valid_keys if k in spam_setup_params}
        set_spam_parameters( **valid_setup_params ) # Missing keys will use default value from function
    
    return fortran_ar

def spam_orbit( collision_param, spam_setup_params = None, current_time = 0.0 ):
    """
    This function is a wrapper for the spam_module.orbit_run function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        spam_setup_params (dict, optional): Dictionary of parameters to set the simulation variables
        current_time (float, optional): Current time of the simulation, for adjusting the times returned by SPAM.

    Returns:
        orbit_path (np.ndarray [x,y,z,vz,vz,t] ): 
            Array of particles indicating the path of the secondary galaxy
            NOTE: The primary galaxy is fixed at origin throughout the simulation.
    """
    LOGGER.info("Running SPAM Simulation - Orbit Run")
    LOGGER.debug("Collision Param: %s" % str(collision_param))
    LOGGER.debug("Current Time: %.2f" % current_time)

    # Check input parameters
    try:
        fortran_ar = initialize_setup( collision_param, spam_setup_params )
    except Exception as e:
        LOGGER.error(f"Failed to initialize setup:\n{e}\n")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error("Setup Params: {spam_setup_params}")
        LOGGER.error(f"Current_time:{current_time}")
        raise ValueError("Failed to initialize setup") from e

    # We need to know how large the final orbit path will be before calling the function
    try:
        LOGGER.debug(f"Calling spam_module.calc_orbit_time_steps")
        n_time_steps = spam_module.calc_orbit_integration_steps( fortran_ar )
    except Exception as e:
        LOGGER.error("Failed to call 'spam_module.calc_orbit_time_steps'")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"Setup Params: {spam_setup_params}")
        LOGGER.error(f"Current_time: {current_time}")
        LOGGER.error(f"Exception: \n",exc_info=True)
        raise ValueError(f"Failed to call 'spam_module.calc_orbit_time_steps'") from e
    
    # Call the Fortran function to create the orbit path
    try:
        LOGGER.debug(f"Calling spam_module.orbit_run.  Time steps: {n_time_steps}")
        orbit_path = spam_module.orbit_run( fortran_ar, n_time_steps )
        LOGGER.debug(f"Returned Orbit Path: {orbit_path.shape}")
    except:
        LOGGER.error("Failed to call 'spam_module.orbit_run'")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"Setup Params: {spam_setup_params}")
        LOGGER.error(f"Current_time: {current_time}")
        LOGGER.error(f"Time Steps: {n_time_steps}")
        LOGGER.error(f"Exception: \n",exc_info=True)
        raise ValueError(f"Failed to call 'spam_module.orbit_run'") from e

    # Adjust the time of the orbit path
    if current_time != 0.0:
        LOGGER.debug(f"Adjusting Orbit Path by Time: {current_time}")
        LOGGER.debug(f"Before Time: {orbit_path[-1,-1]}")
        orbit_path[:,-1] += current_time - orbit_path[-1,-1]
        LOGGER.debug(f"After Time: {orbit_path[-1,-1]}")

    return orbit_path

def basic_run( collision_param, spam_setup_params = None,
              npts1 = 100, npts2 = 50, heat1 = 0.0, heat2 = 0.0, 
              save_dir = None,
            #   save_dir = '/home/mbo2d/galStuff/data/spam_runs/',
            ):
    """
    This function is a wrapper for the spam_module.basic_disk function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        spam_setup_params (dict, optional): Dictionary of parameters to set the simulation variables
        npt1 (int): Number of particles for the primary galaxy
        npt2 (int): Number of particles for the secondary galaxy
        heat1 (float): random motion parameter for the primary galaxy
        heat2 (float): random motion parameter for the secondary galaxy
    
    Returns:
        init_pts (np.ndarray): Array of particles before collision interactions
        final_pts (np.ndarray): Array of particles after collision interactions

    """
    LOGGER.info("Running SPAM Simulation - Basic Run")
    LOGGER.debug("Collision param: %s" % collision_param)
    LOGGER.debug("Number of Particles: %d - %d" % (npts1, npts2))
    LOGGER.debug("Heat Parameters: %f - %f" % (heat1, heat2))

    # Setup up SPAM simulation inputs
    fortran_ar = initialize_setup( collision_param, spam_setup_params )

    # Create unique hex from collision and spam parameters
    if save_dir:

        c_str = ",".join( [f"-{val:.5f}" for val in collision_param] )
        LOGGER.debug("Collision Param String: %s" % c_str)
        
        spam_param = get_spam_parameters() # get all params
        spam_str = ",".join( [f"-{spam_param[k]:.5f}" for k in spam_param] )
        LOGGER.debug("SPAM Param String: %s" % spam_str)

        sim_param = {
            'npts1': npts1,
            'npts2': npts2,
            'heat1': heat1,
            'heat2': heat2,
        }
        sim_str = ",".join( [f"-{sim_param[k]:.5f}" for k in sim_param] )
        LOGGER.debug("Simulation Param String: %s" % sim_str)

        # Combine all strings to create a unique identifier
        total_str = "_".join( [c_str, spam_str, sim_str] )
        LOGGER.debug("Total String: %s" % total_str)
        # Create a unique hex string from the total string

        hex_str = hashlib.md5( total_str.encode() ).hexdigest()
        save_loc = os.path.join( save_dir, f"{hex_str}.npz" )
    
    # Check if the file already exists
    init_pts = None
    final_pts = None

    if save_dir and os.path.exists(save_loc):
        LOGGER.info(f"File already exists. Loading: {save_loc}")
        try:
            LOGGER.info( f"Loading Orbit: {save_loc}" )
            with np.load( save_loc ) as data:
                init_pts = data['init_pts']
                final_pts = data['final_pts']
                LOGGER.debug(f"Loaded SPAM Particles: {init_pts.shape} - {final_pts.shape}")
            return (init_pts, final_pts)

        except:
            pass
    
    # Call the Fortran function
    try:
        LOGGER.debug(f"Calling spam_module.basic_run")
        init_pts, final_pts = spam_module.basic_run( fortran_ar, npts1, npts2, heat1, heat2)
        LOGGER.debug(f"Returned Disk Particles: {init_pts.shape} - {final_pts.shape}")
    except Exception as e:
        LOGGER.error(f"Failed to call 'spam_module.basic_run'")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"npts1 - npts2: {npts1} - {npts2}")
        LOGGER.error(f"heat1 - heat2: {heat1} - {heat2}")
        LOGGER.error(f"Exception: \n",exc_info=True)
        raise ValueError(f"Failed to call 'spam_module.basic_run'") from e

    # If save_dir given, save the initial and final particles to files
    if save_dir:
        LOGGER.info(f"Saving Particles to: {save_loc}")
        np.savez( save_loc, init_pts=init_pts, final_pts=final_pts )
        LOGGER.debug(f"Saved SPAM Particles: {init_pts.shape} - {final_pts.shape}")

    return (init_pts, final_pts)

def run_spam_bundle_1( collision_param, spam_setup_params = None,
                npts1 = 1000, npts2 = 1000, heat1 = 0.1, heat2 = 0.1,
                save_dir = '/home/mbo2d/galStuff/data/spam_runs/'
        ):
    """
    This function is a wrapper for the spam_module.basic_run function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        spam_setup_params (dict, optional): Dictionary of parameters to set the simulation variables
        npt1 (int): Number of particles for the primary galaxy
        npt2 (int): Number of particles for the secondary galaxy
        heat1 (float): random motion parameter for the primary galaxy
        heat2 (float): random motion parameter for the secondary galaxy
        save_loc (str): Save location for the particles
    
    Returns:
        init_pts (np.ndarray): Array of particles before collision interactions
        final_pts (np.ndarray): Array of particles after collision interactions

    """
    LOGGER.info("Running SPAM Simulation - Basic Run")
    LOGGER.debug("Collision param: %s" % collision_param)
    LOGGER.debug("Number of Particles: %d - %d" % (npts1, npts2))
    LOGGER.debug("Heat Parameters: %f - %f" % (heat1, heat2))

    fortran_ar = initialize_setup( collision_param, spam_setup_params )

    # Call the Fortran function
    try:
        LOGGER.debug(f"Calling spam_module.basic_run")
        init_pts, final_pts = spam_module.basic_run( fortran_ar, npts1, npts2, heat1, heat2)
        LOGGER.debug(f"Returned Disk Particles: {init_pts.shape} - {final_pts.shape}")
    except Exception as e:
        LOGGER.error(f"Failed to call 'spam_module.basic_run'")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"npts1 - npts2: {npts1} - {npts2}")
        LOGGER.error(f"heat1 - heat2: {heat1} - {heat2}")
        LOGGER.error(f"Exception: \n",exc_info=True)
        raise ValueError(f"Failed to call 'spam_module.basic_run'") from e

    return (init_pts, final_pts)


def basic_disk( collision_param, spam_setup_params = None, npts1 = 1000, npts2 = 1000):
    """
    This function is a wrapper for the spam_module.basic_disk function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        npt1 (int): Number of particles for the primary galaxy
        npt2 (int): Number of particles for the secondary galaxy
    
    Returns:
        disk_pts (np.ndarray): Array of particles for primary and secondary disk

    """

    LOGGER.info("Running SPAM Simulation - Basic Disk")
    LOGGER.debug(f"Collision Param: {collision_param}")
    LOGGER.debug(f"Number of Particles: {npts1} - {npts2}")

    #initialize the simulation variables
    fortran_ar = initialize_setup( collision_param, spam_setup_params )

    # Call the Fortran function
    try:
        LOGGER.debug(f"Calling spam_module.basic_disk")
        disk_pts = spam_module.basic_disk( fortran_ar, npts1, npts2 )
        LOGGER.debug(f"Returned Disk Particles: {disk_pts.shape}")

    except Exception as e:
        LOGGER.error("Failed to call spam_module.basic_disk")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"Setup Params: {spam_setup_params}")
        LOGGER.error(f"Number of Particles: {npts1} - {npts2}")
        LOGGER.error(f"Exception: \n",exc_info=True)
        raise ValueError("Failed to call spam_module.basic_disk:") from e
    
    LOGGER.debug(f"Returned Disk Particles: {disk_pts.shape}")
    return disk_pts


# ======================== PLOT FUNCTIONS ======================== #

def plot_all_from_param(collision_param, plot_loc=None):
    """
    This function will take collision_parameters and plot several aspects of the simulation

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        plot_loc (str, optional): Location to save the plot

    Returns:
        plot_fig (matplotlib.figure.Figure): Figure with several subplots
    """

    LOGGER.info("Plotting Orbit of a Simulation")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    n1 = 500
    n2 = 500

    # Call functions to get orbit and particle data
    orbit_path = spam_orbit(collision_param)
    init_pts, final_pts = basic_run(collision_param, npts1=n1, npts2=n2)

    # Calculate index of minimum radius to origin (Distance of closest approach)
    min_r_idx = np.argmin(np.linalg.norm(orbit_path, axis=1))

    # Function to calculate the normal vector of the orbit plane
    def plane_normal(points):
        # Calculate the covariance matrix of the points
        cov_matrix = np.cov(points.T)
        # Eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        # The eigenvector with the smallest eigenvalue is the normal vector of the plane
        normal_vector = eigvecs[:, np.argmin(eigvals)]
        return normal_vector

    # Calculate the normal vector of the orbit path
    normal_vector = plane_normal(orbit_path)

    # Convert the normal vector to spherical coordinates for setting the viewing angle
    r = np.linalg.norm(normal_vector)
    elev = np.degrees(np.arccos(normal_vector[2] / r))
    azim = np.degrees(np.arctan2(normal_vector[1], normal_vector[0]))

    # Create figure with top and bottom subplots    # Create figure with top and bottom subplots
    fig = plt.figure(figsize=(6, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[2, 2, 2, 2, 2, ])

    ax1 = fig.add_subplot(gs[:2])
    ax2 = fig.add_subplot(gs[2:], projection='3d')

    # FIRST SUBPLOT: Final particles as seen from perspective
    ax1.scatter(final_pts[0:n1, 0], final_pts[0:n1, 1], s=1, c='b', label='Primary Disk')
    ax1.scatter(final_pts[n1:n1+n2, 0], final_pts[n1:n1+n2, 1], s=1, c='r', label='Secondary Disk')
    ax1.scatter(final_pts[-1, 0], final_pts[-1, 1], s=10, c='k', label=f'Secondary Center: ({final_pts[-1,0]:.2f},{final_pts[-1,1]:.2f})')
    ax1.set_title('Final Particles Perspective')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.set_aspect('equal', 'box')

    # SECOND SUBPLOT: 3D plot with good angle of orbit path
    ax2.scatter(final_pts[::3, 0], final_pts[::3, 1], final_pts[::3, 2], c='r', label='Final')
    ax2.scatter(init_pts[::3, 0], init_pts[::3, 1], init_pts[::3, 2], c='b', label='Initial')
    ax2.plot(orbit_path[:, 0], orbit_path[:, 1], orbit_path[:, 2], c='k', label='Orbit Path')
    ax2.quiver(0, 0, 0, orbit_path[min_r_idx, 0], orbit_path[min_r_idx, 1], orbit_path[min_r_idx, 2], color='k', label='Min Radius Vector')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Orbit Path')
    ax2.legend()
    ax2.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    # If plot_loc is given, save the figure to that location
    if plot_loc:
        plt.savefig(plot_loc)

    return fig


# ======================== MAIN / TESTING ========================== #

if __name__ == '__main__':

    print("\SIMULATOR\n")
    gu.tabprint("This module is intended to be imported and used in other scripts.")
    gu.tabprint("Running script as main will run tests and examples.")

    print("\nInitializing Arguments, and Logger\n")
    try:
        args, LOGGER = gu.initialize_environment()
    except:
        print("Failed to Initalize Arguments and Logger")
        sys.exit(1)

    print( f"\nArgs: \n{args} \nn")
    print( f"\nLogger: \n{LOGGER}\n")
    
    print("Changing LOGGER to Debug")
    gu.change_logging_level('DEBUG')

    print("\nReimport custom modules since Logger changed\n")
    import importlib
    importlib.reload(mm)

    print("\n Testing model_manager (storage_mode = memory)\n")
    test_manager = mm.Model_Manager( model_data=test_param )
    print( f"\nModel_Manger: \n{test_manager}" )
    print( f"\nManager Model Data: \n{test_manager.info['model_data']}" )

    print("\nTesting basic_disk")
    basic_disk( test_param )

    print("\nTesting orbit_run")
    spam_orbit( test_param )

    print("\nTesting Basic Run")
    ipts, fpts = basic_run( test_param )

    print("\nThis SHOULD cause an issue")
    try:
        basic_disk( None )
    except:
        print("Failed as expected")

    print("\nPlotting Example Collision")

    print("Changing LOGGER to INFO becuase matplotlib gives way too many debug messages.")
    gu.change_logging_level('INFO')
    import importlib
    importlib.reload(mm)

    plot_all_from_param( test_param, 'test_orbit_plot.png' )

    print()
    