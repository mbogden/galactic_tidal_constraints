#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
File: tng_find_targets.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2023-Nov-09

Description: 
    This code is designed to interact with the IllustrisTNG Simulation Data and Subhalo catalogs. 
    It's goal is to identify close interactions/mergers between two galaxies.

References:  
- TNG50 Simulation Data
    - Nelson, D. et al. (2015). The Illustris Simulation: Public Data Release. Astronomy and Computing, 13, 12–37. https://doi.org/10.1016/j.ascom.2015.09.003
    - Nelson, D. et al. (2019). First results from the TNG50 simulation: Galactic outflows driven by supernovae and black hole feedback. Monthly Notices of the Royal Astronomical Society, 490(3), 3234–3261. https://doi.org/10.1093/mnras/stz2306
    - Pillepich, A. et al. (2019). First results from the TNG50 simulation: The evolution of stellar and gaseous discs across cosmic time. Monthly Notices of the Royal Astronomical Society, 490(3), 3196–3233. https://doi.org/10.1093/mnras/stz2338
    
- Subhalo Morphology (Deep Learning)
    - Huertas-Company, M.et al. (2019). The Hubble Sequence at $z\sim0$ in the IllustrisTNG simulation with deep learning. Monthly Notices of the Royal Astronomical Society, 489(2), 1859–1879. https://doi.org/10.1093/mnras/stz2191
    - Varma, S., Huertas-Company, M., Pillepich, A., Nelson, D., Rodriguez-Gomez, V., Dekel, A., Faber, S. M., Iglesias-Navarro, P., Koo, D. C., & Primack, J. (2021). The building up of observed stellar scaling relations of massive galaxies and the connection to black hole growth in the TNG50 simulation. Monthly Notices of the Royal Astronomical Society, 509(2), 2654–2673. https://doi.org/10.1093/mnras/stab3149
    
Acknowledgements:    
- Sections of this code were written with the assistance of ChatGPT made by OpenAI.

- The IllustrisTNG simulations were undertaken with compute time awarded by 
    the Gauss Centre for Supercomputing (GCS) under GCS Large-Scale Projects GCS-ILLU 
    and GCS-DWAR on the GCS share of the supercomputer Hazel Hen 
    at the High Performance Computing Center Stuttgart (HLRS), 
    as well as on the machines of the Max Planck Computing and Data Facility (MPCDF) in Garching, Germany.
"""


# # Finding Galaxy Mergers within the IllustrisTNG Simulation

# ## Imports

# In[2]:


# ================================ IMPORTS ================================ #
import os, argparse, h5py
import numpy as np, pandas as pd, scipy.signal
import matplotlib.pyplot as plt 
import illustris_python as il
import tng_functions as tf
from collections import Counter

print("Imports Done")

# Global variables
SIM_DIR = '/home/tnguser/sims.TNG/TNG50-1/output/'

# A useful fucntion I often use for indented printing
def tabprint( printme, start = '\t - ', end = '\n' ):
    print( start + str(printme), end = end )


# ---
# ## Command Line Arguments
# 
# This is written in JupyterLab, and will be compiled and ran in python for faster execution.  This will define the possible input command line arguements.
# 
# 
# WARNING:  I have not been consistent with implementing and following arguments.  Code still in indevlopment.  

# In[3]:


# This argument decides if code is in python or jupyter.
buildEnv = False

# Define argument parser function 
def initParser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument( '-s', '--simDir', default = '/home/tnguser/sims.TNG/TNG50-1/output/',  type=str, \
                        help="Base directory for a single simulation on the IllustrisTNG servers.")   
    
    parser.add_argument( '-n', '--simName', default = 'TNG50-1',  type=str, \
                        help="Name for the simulation being worked on.")
    
    parser.add_argument( '-o', '--overwrite', default = False,  type=bool, \
                        help="Overwrite output files?  If false, will check if output file exists before beginning time-consuming tasks.")
    
    parser.add_argument( '-t', '--trim', default = -1,  type=int, \
                        help="Default number of subhalos to consider, sorted by highest mass first.")
    
    parser.add_argument( '-f', '--function', default = 'None', type=str, \
                        help="Default function program will be executing.")
    
    parser.add_argument( '-d', '--dataDir', default = 'data', type=str, \
                        help="Default location to store misc data files.")

    return parser

parser = initParser()
print("Args: Defined")


# ## To Python? Or to JupyterLab? 
# This will establish if this is being run in a JupyterLab environment or from Command Line in Python. 
# 
# NOTE:  If you're running this in Jupyter, modify the `cmdStr` below to whatever variables you need.

# In[4]:


# Am I in a jupyter notebook?
try:
    
    # This command is NOT available in a python script
    get_ipython().__class__.__name__
    buildEnv = True
    print ("In Building Environment")
    
    # Command Line Arguments
    cmdStr  = 'python3 targets-working.py'
    cmdStr += ' --trim 10'
    cmdStr += ' --dataDir tng-data'
    
    # Read string as if command line
    print( "CMD Line: \n\t$:", cmdStr)
    
    # This function doesn't like the 'python3 file.py' part.
    args = parser.parse_args(cmdStr.split()[2:])

# Or am I in a python script?
except:
    
    # Read CMD arguments
    args = parser.parse_args()
    

print( "Args: Read")
print( args )

if args.dataDir[-1] != '/': args.dataDir += '/'

# Setup data directory if not found
os.makedirs(args.dataDir, exist_ok=True)
os.makedirs(args.dataDir + 'tmp', exist_ok=True)


# In[5]:


if buildEnv: 
    # Location of one simulation
    print("Is this locational valid?")
    print( f"Simulation data: {os.path.exists( args.simDir )} - {args.simDir}" )


# ---
# # Halos and SubHalos
# Within the simulation, Halos are the largest set of objects that are gravitationally bound to each other, I like to think of them as galaxy clusters.  Subhalos are also gravitationally bound objects but more dense, and I suspect has to do with potential energy.   I like to think of them as galaxies, globular clusters, blobs of intergalactic gas, etc.  (That's my reasoning and I'm sticking to it)
# 
# 
# For more information, pleas visit the IllustrisTNG Data Specification Page.  https://www.tng-project.org/data/docs/specifications/

# ---
# ## Mass Filter
# 
# So I am looking for galaxies that have enough mass to visualize well.  Thus I will be choosing galaxies that are a scale multiple above and below the mass of the Milky Way.

# In[6]:


def getMassFilter( args, snapNum, mScale = 10 ):
    
    # Define where file will be saved
    mLoc = f'{args.dataDir}/tmp/{args.simName}-{snapNum}-mask-mass-{mScale}.npy'
    
    # Read from file if it exits
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Mass Mask: {mLoc}")
        mass_mask = np.load( mLoc )
        return mass_mask
    
    # define mass limits
    milky_way_mass = 150.0  # in (10^10 M_⊙) 
    upper_mass = milky_way_mass * mScale
    lower_mass = milky_way_mass / mScale
    
    # Pull masses for all subhalos in snapshot
    fields = ['SubhaloMass']
    print("Pulling Masses for all Subhalos")
    print("WARNING: May take a while ")
    SubhaloMass = il.groupcat.loadSubhalos( args.simDir, snapNum, fields=fields)
    
    # This is the first occasion where I wi
    
    # Find galaxies between upper and lower mass
    mask_mass = ( SubhaloMass[:] <= upper_mass ) & ( SubhaloMass[:] >= lower_mass )
    
    # Save mass
    np.save( mLoc, mask_mass )
    
    return mask_mass
    
if buildEnv and True:
    args.overwrite = False
    mask_mass = getMassFilter( args, 50 )
    print( mask_mass.shape, mask_mass.dtype )
    mask_mass = getMassFilter( args, 67 )
    print( mask_mass.shape, mask_mass.dtype )


# ___
# ## Centrals and Satellites
# Halo's often have a central largest galaxy, with smaller subhalos in orbit around the halo called satellites.  For convenience, let's create a mask of these central galaxies.
# 

# In[7]:


def expand_mask_from_list( true_list, snapnum ):    
    n_subhalo = il.groupcat.loadHeader( args.simDir, snapnum)['Nsubgroups_Total']
    mask = np.full( n_subhalo, False, dtype=bool )
    mask[true_list] = True    
    return mask
    

def getCentralFilter( args, snapnum = 99 ):
    
    mLoc = f'{args.dataDir}/tmp/{args.simName}-{snapnum}-mask-central.npy'

    # If already obtained, read from file
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Central Galaxy file: {mLoc}")
        mask_central = np.load( mLoc )
        return mask_central

    print(f"Getting Central SubHalo IDs for sim/snapshot: {args.simName} / {snapnum}")

    # The GroupFirstSub is the subhalo id for the largest subhalo in a halo.  
    GroupFirstSub = il.groupcat.loadHalos( args.simDir, snapnum, fields=['GroupFirstSub'])

    # Filter out groups that contain no subhalos.
    w = np.where(GroupFirstSub >= 0) # value of -1 indicates no subhalo in this group
    central_ids = GroupFirstSub[w]
    
    # Expand into a full array with a value for every subhalo
    mask_central = expand_mask_from_list( central_ids, snapnum = snapnum )
    
    # Save mass
    np.save( mLoc, mask_central )
    
    return mask_central

if buildEnv and True: 

    args.overwrite = False
    mask_central = getCentralFilter( args, snapnum = 50 )
    print('Central Galaxies:', mask_central.shape, mask_central[:10] )
    mask_central = getCentralFilter( args, snapnum = 67 )
    print('Central Galaxies:', mask_central.shape, mask_central[:10] )
    


# In[8]:


if buildEnv:
    tmp2 = il.groupcat.loadHeader(args.simDir, 67)
    print( tmp2 )
    print( tmp2['Nsubgroups_Total'] )
    print( type(tmp2['Nsubgroups_Total'] ))


# # (r) Galaxy Morphologies (Deep Learning)
# 
# Because our method relies on disks of galaxies, it might be useful for us to find mergers betweeen two disk galaxies. 

# In[9]:


def getDiskMorphologyFilter( args, snapNum = 99 ):
    
    mLoc = f'{args.dataDir}/tmp/{args.simName}-{snapNum}-mask-disk-morphology.npy'

    # If already obtained, read from file
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Disk Morphology Mask: {mLoc}")
        mask_disk = np.load( mLoc )
        return mask_disk
    
    # Check if morphology file exists.  
    hdf5Loc = f'subcatalogs/TNG50-1-morphologies_deeplearn.hdf5'
    if not os.path.exists( hdf5Loc ):
        print("WARNING:!  Subcatalog File missing: ", hdf5Loc )
        print("Please see IllustrisTNG Data Specs for info to download this file.")
        raise AssertionError
    
    # Read the deeplearning morphology file
    with h5py.File(f'subcatalogs/TNG50-1-morphologies_deeplearn.hdf5', 'r') as file:
        
        header = f'Snapshot_{snapNum}'
        
        # Verify snapshot header is in file
        if header not in file.keys():
            print(f"Bad HDF5 header: {header} / {file.keys()}" )
            return None       
        
        subhaloIDs      = np.array( file[header]['SubhaloID'] )
        subhaloDiskProb = np.array( file[header]['P_Disk'] )
        
        print( 'test morph: ', subhaloIDs.shape )
        
        # Iterate through and grab subhalos with a greater chance of being a disk galaxy
        disk_list = []
        for i in range( subhaloIDs.shape[0] ):
            if subhaloDiskProb[i] > 0.5:
                disk_list.append( subhaloIDs[i] )
        
    # Done reading file.
    
    # create mask 
    mask_disk = expand_mask_from_list( np.array( disk_list ), snapNum )
    
    # Save mass
    print(f"Saving Disk Morphology Mask: {mLoc}")
    np.save( mLoc, mask_disk )
    
    return mask_disk    


if buildEnv and True:
    tf.explore_hdf5( 'subcatalogs/TNG50-1-morphologies_deeplearn.hdf5', max_depth = 0 )
    # args.overwrite=False
    # mask_disk = getDiskMorphologyFilter( args, 67 )  
    # print('Disk Galaxies:', mask_disk.shape, mask_disk[:10] )
    # mask_disk = getDiskMorphologyFilter( args, 50 )  
    # print('Disk Galaxies:', mask_disk.shape, mask_disk[:10] )


# ## (y) Merger History
# 
# Because manually detecting major mergers in the merger tree is messy (trust me, I tried), I'll be using someone else's subcatalogs to detect major mergers between galaxies.   

# In[10]:


def getMajorMergerMask( args, snapNum = 67, snapCutoff=13 ):
    
    mLoc = f'{args.dataDir}/tmp/{args.simName}-{snapNum}-mask-major-merger-{snapCutoff}.npy'
    
    # If already obtained, read from file
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Upcoming Major Merger Mask: {mLoc}")
        mask_merger = np.load( mLoc )
        return mask_merger
    
    file_loc = f'subcatalogs/MergerHistory_0{snapNum}.hdf5'
    
    print( f"Merger History Loc: {file_loc}" )
    
    # Return None if no file found for snap num.
    if not os.path.exists( file_loc ):
        print(f"WARNING:  Could not find file: {file_loc}")
        raise ValueError(f"Subcatalog File Missing: {file_loc}")
        return None
    
    # Read Merger History file.    
    with h5py.File(file_loc, 'r') as file:
            
        # Get the object (could be a group or dataset)
        dataset = file['SnapNumNextMajorMerger']
        
        # Create a boolean mask for values that are non-negative and below the upper limit
        mask_merger = (dataset[:] >= 0) & (dataset[:] <= (snapNum + snapCutoff) )
        
    # Saving
    print(f"Saving Major Merger Mask: {mLoc}")
    np.save( mLoc, mask_merger )
       
    return mask_merger
            
    # Find merger
    
# Get merger tree catalog


if buildEnv and True:
    
    # print_HDF5_info( f'subcatalog/MergerHistory_0{snapNum}.hdf5' )
    
    mask_merger = getMajorMergerMask( args, 50, 18 )
    print('Disk Galaxies:', mask_merger.shape, mask_merger[:10] )
    mask_merger = getMajorMergerMask( args, 67, 33 )
    
    print('Disk Galaxies:', mask_merger.shape, mask_merger[:10] )
    
        


# ---
# ## Combine Masks and Generate MOI_1 IDs
# 
# (Matt from the future here) 
# This is step one of many to narrow down potential mergers so I'm going to establish some terminology as I'm rewriting this.  potential mergers will hence forth be furthered as Mergers-of-Interest or MOI for short.  And each step will have a number, so this will be moi_1.  
# 
# ### MOI_1: Parent galaxy that will collide soon.

# In[11]:


def combine_masks(mask_list):
    
    # Verify that all masks have the same shape
    if not all(mask.shape == mask_list[0].shape for mask in mask_list):
        print("ERROR:  Masks do not have the same shape.")
        for mask in mask_list:
            print( len(mask) )
        raise ValueError("ERROR: Combine Masks: All masks must have the same shape")

    # Initialize the combined mask with the first mask
    combined_mask = mask_list[0].copy()

    # Perform logical AND operation with each subsequent mask
    for mask in mask_list[1:]:
        combined_mask &= mask

    return combined_mask

def generate_moi_1_ids( args, snapNum, mass = True, massScale = 10, central = False, disk = True, major = True, majorCutoff = 13 ):
    
    # Create list of mask to find goi
    mask_list = []
    if mass:     mask_list.append( getMassFilter           ( args, snapNum, mScale = massScale ) )
    if central:  mask_list.append( getCentralFilter        ( args, snapNum ) )
    if disk:     mask_list.append( getDiskMorphologyFilter ( args, snapNum ) )
    if major:    mask_list.append( getMajorMergerMask      ( args, snapNum, majorCutoff ) )

    try:
        # Get 
        combined_mask = combine_masks( mask_list )
        subhalo_ids = np.where( combined_mask )[0] 
        
    except ValueError as e:
        print(e)
        return None
    
    # Loop through subhalo ids and combine with snapnum to create SubhaloIDRaw
    moi_1_ids = []
    
    for mid in subhalo_ids:
        moi_1_ids.append( tf.generate_subhalo_id_raw( snapNum, mid, )  )
    
    moi_1_ids = np.array( moi_1_ids )
    
    return moi_1_ids


    
    
if buildEnv and True:
    

    # Starting snaps and their ranges for looking
    #  (based on merger history hdf5 file)
    snap_info = { 
        25: {'start_snap':25, 'stop_snap':29}, 
        29: {'start_snap':29, 'stop_snap':33}, 
        33: {'start_snap':33, 'stop_snap':40},
        40: {'start_snap':40, 'stop_snap':55}, 
        50: {'start_snap':50, 'stop_snap':67}, 
        67: {'start_snap':67, 'stop_snap':99}
    }
    
    moi_1_ids = []
    args.overwrite = False
        
    for snap in snap_info:

        # Initial GOIs of interest
        snap_info[snap]['moi_1'] = generate_moi_1_ids( args, snap, majorCutoff = snap_info[snap]['stop_snap']- snap_info[snap]['start_snap']) 
    
    c = 0
    for s in snap_info:
        c += len( snap_info[s]['moi_1'] )
    
    print("done")
    print( "MOI_1 Found: ", c )


# --- 
# ## Find MOI_2 in Future Merger Trees
# 
# Since requesting a merger tree only returns it's tree for the current moment and backwards in time, I need to jump several snapshots forward, and identify which galaxies it belongs to there.  It's a long tedious process but I'll figure it out.
# 
# ### MOI_2:  Child galaxy that has already undergone a major merger event

# In[12]:


def find_mois_in_tree( tree_id_raw, moi_1_list, args, ):
    
    tree_snap, tree_id = tf.deconstruct_subhalo_id_raw( tree_id_raw )

    # Load only SubhaloIDRaw for effecient retrieval time
    tree = il.sublink.loadTree( args.simDir, tree_snap, tree_id, fields=['SubhaloIDRaw'] )

    # See if any of my MOIs are in list
    moi_1_mask = np.isin( moi_1_list, tree )
    n_matches = np.sum(moi_1_mask) # Presuming True = 1 and False = 0

    # If none found
    if n_matches == 0:
        return []
    # Else we have results

    # Get the index locations where the mask is True
    moi_1_loc = np.where(moi_1_mask)[0]
    moi_1_ids_in_tree = moi_1_list[ moi_1_loc ]

    moi_2_list = []

    for moi_1 in moi_1_ids_in_tree:
        moi_2_list.append( ( tree_id_raw, moi_1 ) )

    return moi_2_list


def getMOI_2( args, snap_info, func_overwrite = False):
    
    # Loop through snaps of interest, collect moi_2 value
    
    for snap in snap_info:
        
        print(f"Snap {snap} of {snap_info.keys()}")
        stop = snap_info[snap]['stop_snap']
        moi_1_list = snap_info[snap]['moi_1']
        
            
        # Was tree already exist on disk?  (It takes a while to load 1000s)
        m2Loc = f'{args.dataDir}/tmp/{args.simName}-moi_2-{stop}.npy'

        # If already obtained, read from file
        if os.path.exists( m2Loc ) and not args.overwrite and not func_overwrite:
            print(f"Reading Moi_2 results: {m2Loc}")
            snap_info[snap]['moi_2'] = np.load( m2Loc )
            continue
    
        # Basically just get lots of galaxies above mass range in final snapshot
        tree_id_list = generate_moi_1_ids( args, stop, \
                                               mass = True, massScale = 20, \
                                               central = False, disk = False, \
                                               major=False )
        # Find moi_2 containing moi_1
        moi_2_list = []
        for i, tree_id_raw in enumerate(tree_id_list):  
            tree_snap, tree_id = tf.deconstruct_subhalo_id_raw( tree_id_raw )
            tabprint( f" {i} / {tree_id_list.shape[0]} - {tree_id_raw} - Found: {len(moi_2_list)}", end='\r' )

            found_ids = find_mois_in_tree( tree_id_raw, moi_1_list, args )
            moi_2_list.extend( found_ids )

        print( f"\nFound MOI_2 / Tree Matches: {len( moi_2_list) }")
        
        np.save( m2Loc, moi_2_list )        
        snap_info[snap]['moi_2'] = moi_2_list

    return snap_info

if buildEnv and True:
    
    args.overwrite = False
    snap_info = getMOI_2( args, snap_info )
    test_m2_list = snap_info[67]['moi_2'][:10]
    
    print( type( snap_info ) )
    print( test_m2_list )
        


# In[13]:


# Define print fucntion for a row
def printRow( tree, i, fields ):
    # if i == -1:
    #     print("Invalid index")
    #     return
    
    print( " - ".join( [ f"{key}:{tree[key][i]}" for key in fields ]) )
    
def createVisLink( subhaloIDRaw, projection = 'face', simulation='TNG50-1' ):
        tmp = tf.deconstruct_subhalo_id_raw( subhaloIDRaw )    
        
        if projection == 'face':
            link = f"https://www.tng-project.org/api/{simulation}/snapshots/{tmp[0]}/subhalos/{tmp[1]}/vis.png?partType=stars&partField=stellarComp-jwst_f200w-jwst_f115w-jwst_f070w&size=1&method=histo&rotation=face-on&plotStyle=edged"
        # Else, project x,y plane
        else:
            link = f"https://www.tng-project.org/api/{simulation}/snapshots/{tmp[0]}/subhalos/{tmp[1]}/vis.png?partType=stars&partField=stellarComp-jwst_f200w-jwst_f115w-jwst_f070w&size=1&method=histo&nPixels=256%2C256&axes=0%2C1&plotStyle=edged"
        return link

def get_merger_snapshot( id_raw ):
    
    snap, subhalo_id = tf.deconstruct_subhalo_id_raw( id_raw )
    
    merger_loc = f'subcatalogs/MergerHistory_0{snap}.hdf5'
    
    # Return None if no file found for snap num.
    if not os.path.exists( merger_loc ):
        print(f"WARNING:  Could not find file: {merger_loc}")
        raise ValueError(f"Subcatalog File Missing: {merger_loc}")
        return None
    
    # Read Merger History file.    
    with h5py.File(merger_loc, 'r') as file:
            
        # Get the object (could be a group or dataset)
        dataset = file['SnapNumNextMajorMerger']
        
        merger_snap = dataset[subhalo_id]
        
    
    return merger_snap

if buildEnv:
    tmp = get_merger_snapshot( test_m2_list[0][1] )
    print(' Snap: ', tmp )


# ## MOI_3:  MOI_2 with multiple MOI_1 in their tree
# 
# Given I have so many targets to choose from, let's narrow it down autmomatically again.  If a tree contains 2+ MOI_1 ids, then there's a good chance both parents of the major merger will meet our desired MOI_1 requirements.  Whereas non-repeats, will only have one parent meeting our requirements. 

# In[14]:


def find_moi_2_repeats( moi_list ):
    
    # Step 1: Extract all moi_2's from the tuples
    moi_2_list = [mois[0] for mois in moi_list]

    # Step 2: Count the occurrences of each moi_2
    moi_2_counts = Counter(moi_2_list)
    
    # Let's see if any contain 3 or more....
    moi_2_3plus = [mois for mois in moi_list if moi_2_counts[mois[0]] >= 3]
    print( "These Mergers have more than 2 repeats, thus they may encounter multiple mergers in the timeframe. Ignoring for now..." )
    print( moi_2_3plus )
    
    # Step 3: Combine MOI_1's into a single row.  
    # SubhaloIDs were given based on descending mass, thus lower numbers should have more mass.  
    # Greater mass parents will be labels as primary, and the other secondary. 
    # The following is magic written by ChatGPT.
    
    # Creating a DataFrame from the list of tuples
    df = pd.DataFrame(moi_list, columns=['moi_2', 'moi_1'])

    # Grouping by 'moi_2' and filtering for pairs
    pairs = df.groupby('moi_2').filter(lambda x: len(x) == 2)

    # Function to assign labels based on the smaller value
    def assign_labels(group):
        smaller_row = group.loc[group['moi_1'].idxmin()]
        larger_row = group.loc[group['moi_1'].idxmax()]

        return pd.Series({
            'moi_2': smaller_row['moi_2'],
            'p_moi_1': smaller_row['moi_1'],
            's_moi_1': larger_row['moi_1']
        })

    # Applying the function to each group
    moi_3 = pairs.groupby('moi_2').apply(assign_labels).reset_index(drop=True)
        
    return moi_3


if buildEnv and True:
    
    args.overwrite = False
    
    # Combine all moi_1/moi_2 lists
    m2_all_list = []
    
    for s in snap_info:
        print( s )
        m2_all_list.extend( snap_info[s]['moi_2'] )
    
    m3_id_df = find_moi_2_repeats( m2_all_list )
    
    print( f"M2 Repeats: {m3_id_df.shape} ")
    print(m3_id_df[:2])


# In[16]:


# NOTE: This loads the MOI_2 Tree for moi_3's.
def travel_tree( args, m3 ):
    
    # Rename common variables
    m2 = m3['moi_2']
    p_m1 = m3['p_moi_1']
    s_m1 = m3['s_moi_1']
    m2_snap, m2_subhalo = tf.deconstruct_subhalo_id_raw( m2 )
    m1_snap, p_m1_subhalo = tf.deconstruct_subhalo_id_raw( p_m1 )
    m1_snap, s_m1_subhalo = tf.deconstruct_subhalo_id_raw( s_m1 )
    
    # Get snapshot when moi is supposed to occur
    merger_snap = get_merger_snapshot( p_m1 )
        
    # Fields to load from Tree
    fields = ['SubhaloID','NextProgenitorID','MainLeafProgenitorID','FirstProgenitorID','SubhaloGrNr',\
              'SubhaloIDRaw','SubhaloMass', 'RootDescendantID', 'SnapNum', 'DescendantID',\
              'SubhaloPos', 'SubhaloVel', 'SubhaloSpin', 'SubhaloHalfmassRad', 'SubhaloCM', ]
    
    # NOTE: Descendent ID is children id in future.
    # NOTE: First ProgenitorID is ID of primary Parent from past. 
    
    
    # Load Merger Tree with desired fields
    tree = il.sublink.loadTree( args.simDir, m2_snap, m2_subhalo, fields=fields)
    
    # Create a dictionary to map Subhalo IDs to their index in the list
    tree_index = {subhalo_id: index for index, subhalo_id in enumerate(tree['SubhaloID'])}  
    
            
    # List to store data in.
    data_list = []
    
    # Define function for extracting row of tree
    def extract_data( snap, p_id, s_id ):
        
        # Store values in a flat dictionary
        flat_dict = {}
        flat_dict['moi_2'] = m2
        flat_dict['snap'] = snap
        flat_dict['merger_snap'] = merger_snap
        flat_dict['p_moi_1'] = p_m1
        flat_dict['s_moi_1'] = s_m1
        flat_dict['xy_projection'] = createVisLink( tree['SubhaloIDRaw'][p_id], projection = 'xy' )
        flat_dict['p_face_projection'] = createVisLink( tree['SubhaloIDRaw'][p_id], projection = 'face' )
        flat_dict['s_face_projection'] = createVisLink( tree['SubhaloIDRaw'][s_id], projection = 'face' )
        
        # Loop through tree fields and store
        for k in fields:
            for c, ii in [ ('p',p_id), ('s',s_id) ]:
                flat_dict[f'{c}_{k}'] = tree[k][ii]
        
        return flat_dict
    
    # Grab index values for both moi_1.
    p_m1_id = np.where( tree['SubhaloIDRaw'] == p_m1  )[0][0]
    s_m1_id = np.where( tree['SubhaloIDRaw'] == s_m1  )[0][0]
    
    # Go backwards a few snapshots first just in case of immidiate merger
    back_snaps = 10
    snap = m1_snap
    p_id = p_m1_id
    s_id = s_m1_id
        
    while snap >= m1_snap - back_snaps:
        
        # Values needed for other fields
        snap -= 1
        p_id = tree_index[tree['FirstProgenitorID'][p_id]]
        s_id = tree_index[tree['FirstProgenitorID'][s_id]]
        
        # Verify the snaps all match
        assert snap == tree['SnapNum'][p_id]
        assert snap == tree['SnapNum'][s_id]
        
        data_list.append( extract_data( snap, p_id, s_id ) )
        
    # Now go forwards in time to grab info
    snap = m1_snap
    p_id = p_m1_id
    s_id = s_m1_id
    
    # Loop until primary and secondary merger, or you've reached the origin of the tree.
    while (p_id != s_id) and snap < m2_snap:
        
        # Add data of interest
        data_list.append( extract_data( snap, p_id, s_id ) )
        
        # Iterate forwards in time
        snap += 1
        p_id = tree_index[tree['DescendantID'][p_id]]
        s_id = tree_index[tree['DescendantID'][s_id]]
        
        # Verify the snaps all match
        if snap != tree['SnapNum'][p_id]: break
        if snap != tree['SnapNum'][s_id]: break
        
    # Convert to dataframe and return    
    return pd.DataFrame(data_list)
        

def create_moi_3_file( args, m3_id_df, file_name_loc = 'tng-data/moi_3', func_overwrite=False ):
        
    # If file exists, read and return.
    if os.path.exists( file_name_loc + '.pkl' ) and args.overwrite == False and func_overwrite == False:
        m3_df = pd.read_pickle( file_name_loc + '.pkl' )
        return m3_df
    
    # Else, create file by getting info via merger trees.
    m3_df_list = []
    for i, row in m3_id_df.iterrows():
        
        print(f"Loading Trees: {i} of {m3_id_df.shape[0]}", end='\r')
        m3_df_list.append( travel_tree(args, row) )
        
        # if i > 2: break
    
    print(f"\nLoading Trees Complete: {m3_id_df.shape[0]}" )
        
    print('')
    print("Final MOI 3 DataFrame")
    # Combine all the dataframes
    m3_df = pd.concat( m3_df_list, ignore_index=True )
    
    # Sort by Moi_2, then by snap
    m3_df = m3_df.sort_values(by=['moi_2', 'snap'], ascending=[True, True])
    
    m3_df.to_csv( file_name_loc + '.csv', index=False )
    m3_df.to_pickle( file_name_loc + '.pkl' )
    
    return m3_df


if buildEnv and True:
    
    args.overwrite = False
    
    ex_df = travel_tree( args, m3_id_df.loc[0] )
    # m3_df = save_moi_info( args, m2_repeats, 'tng-data/moi-2-repeats.csv' )
    print( 'Example Tree: ', ex_df.shape ) 
    print( type( ex_df.loc[0]['p_SubhaloPos'] ) )
    print( ex_df[:2] )
    
    m3_df = create_moi_3_file( args, m3_id_df, file_name_loc = 'tng-data/moi-3', func_overwrite=True )

    print( m3_df.shape )
    


# # MOI_4:   Galaxies that show hints of tidal features.
# 
# While it would be nice if there was an automatic way to review these potential targets for tidal features, I need to find a handful of visually nice targets now.  So I manually opened the links to visuallize the galaxies.  Any that showed hints of tidal features, I'm copying the snapnum and subhalo id down below.
# 
# NOTE to myself.  The following is only from reviewing moi_2_50_67_repeats.   More are sure to be found in newer file moi_3 file.

# In[18]:


moi_4_great = {
    54:45004,
    61:186579,
    62:190717,   # AMAZING
    63:192216,
    64:153811, 
    62:187573,
}

moi_4_good = [
    (54, 45004),
    (55, 46172),
    (56, 46470),
    (62, 187573),
    (63, 189019),
    (64, 205294),
    (61, 186579),
    (62, 190717),   # AMAZING
    (63, 192216),
    (64, 153811),
    (56, 230959),
]


moi_4_maybe = [
    (58, 136813),
    (59, 138897),
    (56, 192189),
    (59, 210457),
    (62, 338866),
    (63, 336934),
    (64, 338830),
    (65, 345132),
    (55, 330164),
    (56, 333739),
    (57, 336505),
]

if buildEnv:
    
    # Let's save some of these test moi-3s somewhere.
    m4_loc = 'tng-data/moi_4-test.pkl'
    m4_csv_loc = m4_loc[:-4] + '.csv'
    
    # Identify MOI_3 id's that contain our MOI_4 id.
    pid_list = [ tf.generate_subhalo_id_raw( snap, subhalo_id ) for snap, subhalo_id in moi_4_good ]
    print( pid_list )
    
    # Extract data associated with moi_3 above.
    moi3_ids = m3_df[ m3_df['p_SubhaloIDRaw'].isin(pid_list) ]['moi_2'].unique()
    print( moi3_ids )
    
    # Get all rows with moi_4 values, thus getting all info before and after moi_4 snap
    moi_4_df = m3_df[ m3_df['moi_2'].isin(moi3_ids) ]
    
    # Look for duplicates
    print(f"MOI_4 Test Shape: {moi_4_df.shape}")
    
    # Add new columns indicating row of MOI_4 candidates
    moi_4_df['moi_4'] = False
    moi_4_df.loc[moi_4_df['p_SubhaloIDRaw'].isin(pid_list), 'moi_4'] = True
    
    moi_4_df.to_pickle( m4_loc )
    moi_4_df.to_csv( m4_csv_loc , index=False )
    


# In[ ]:


#


# In[ ]:




