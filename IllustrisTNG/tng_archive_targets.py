#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: tng-archive-targets.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2026-01-04

Description: Using my previously written code, I will take the best 
    TNG-50 targets I've found, update their partical ownership and 
    galactic dynamics, and save the new particles to be used on babbage. 

References:  Sections of this code were written with the assistance of ChatGPT made by OpenAI.

"""
# ================================ IMPORTS ================================ #

import argparse, h5py, json, math, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

import galaxyJSPAM.IllustrisTNG.tng_functions as tf
import galaxyJSPAM.IllustrisTNG.tng_images as tng_images
import galaxyJSPAM.IllustrisTNG.tng_particles as tng_particles
# !pip install illustris-python
# import IllustrisTNG.tng_images as ti


# Set NumPy print options to reduce the precision to 3 decimal places
np.set_printoptions(precision=2)
buildEnv = True

print("Import Done")


# In[2]:


# ================================ INPUT ARGUMENTS ================================ #

# Define argument parser function 
def initParser():
    
    parser = argparse.ArgumentParser()
    
    # Simulation Arguments
    parser.add_argument( '-s', '--simDir', default = '/home/tnguser/sims.TNG/TNG50-1/output',  type=str, \
                        help="Base directory for a single simulation on the IllustrisTNG servers.")   
    
    parser.add_argument( '-n', '--simName', default = 'TNG50-1',  type=str, \
                        help="Name for the simulation being worked on.")    

    # Function arguments
    parser.add_argument( '-f', '--function', default = 'None', type=str, \
                        help="Unique functions this program can execute from command line.")
    
    parser.add_argument( '-d', '--dataDir', default = 'tng-data', type=str, \
                        help="Default location to store misc data files.")
    
    parser.add_argument('-ow', '--overwrite', action='store_true',
                        help="Overwrite output files? \
                        If false, will check if output file exists before beginning time-consuming tasks.")
    
    parser.add_argument( '-i', '--inputFile', type=str, \
                        help="Input file location.")
    
    parser.add_argument( '-id', '--targetID', type=int, \
                        help="Misc ID value.  Different between functions.")
    
    parser.add_argument( '-o', '--outputFile', type=str, \
                        help="Output file location.")

    return parser

parser = initParser()
script_args = parser.parse_args()
print("Args: Defined")


# In[3]:


if buildEnv:
    
    # Read potential tng targets file
    m4_loc = 'tng-targets/moi-4-targets.pkl'
    assert os.path.exists( m4_loc ), f'Target information file not found: {m4_loc}'
    print(f"Reading: {m4_loc}")    
    m4_df = pd.read_pickle( m4_loc )
    m4_df['p_SubhaloIDRaw'] = m4_df['p_SubhaloIDRaw'].astype(np.uint64)
    m4_df['s_SubhaloIDRaw'] = m4_df['s_SubhaloIDRaw'].astype(np.uint64)
    print( f"M4-DF: {m4_df.shape}")
    
    top_df = m4_df[ m4_df['moi_4_rating'] >= 4 ]
    print( f"Top DF: {top_df.shape}" )
    
    t_id = 62000000190717
    t_row = top_df[ top_df['p_SubhaloIDRaw'] == t_id ]
    
    if len( t_row ) != 1:
        raise ValueError("BAD" )
    t_row = t_row.iloc[0]    
    print("Target Row:")
    print(t_row)
    


# In[ ]:


def create_target_archive( t_row, t_df, data_dir = 'tng-data', overwrite = False, dev_print = False ):
    
    # Make sure it's only for this merger
    t_df = t_df[ t_df['moi_2'] == t_row['moi_2'] ]
    
    # Pull commonly used values
    p_id = t_row['p_SubhaloIDRaw']
    s_id = t_row['s_SubhaloIDRaw']
    t_snap = t_row['snap']
    pos_mat = t_row['orbital_frame_pos_transform']
    vel_mat = t_row['orbital_frame_vel_transform']
    
    if dev_print:
        print( f"P_ID: {p_id}")
    
    # Open Archive File
    archive_loc = os.path.abspath( os.path.join(data_dir, f"target-archive-{p_id}.h5" ) )
    os.makedirs(os.path.dirname(archive_loc), exist_ok=True)
    print( f"Target Archive Loc: ({os.path.exists(archive_loc)}) - {archive_loc}") 
    
    archive = h5py.File( archive_loc, 'a' )
    print( f"Archive File: {type(archive)}")
    
    # hardcode misc attributes
    pt_headers = ["x","y","z","vx","vy","vz","m"]
    pt_units   = ["kpc","kpc","kpc","km_per_s","km_per_s","km_per_s","1e10_Msun"]
    meta_headers = ["particle_type", "raw_subhalo", "hist_subhalo"]
    pt_type_key = json.dumps( {"gas":0, "dark_matter":1, "stars":4, "black_hole":5} )
    
    # Generate Orbital Frame Particles if not exist    
    with h5py.File(archive_loc, "a") as archive:
         
        # Store basic info
        archive.attrs["p_id"]   = int(p_id)
        archive.attrs["s_id"]   = int(s_id)
        archive.attrs["t_snap"] = int(t_snap)
        
        # Dataset Names
        dname_pts  = "particles_orf"
        dname_meta = "particles_meta"
        
        # Check if information exists
        if (dname_pts in archive) and (dname_meta in archive) and not overwrite:
            print("Found particles/meta in archive; skipping (overwrite=False).")
            return
        
        if overwrite:
            if dname_pts in archive:  del archive[dname_pts]
            if dname_meta in archive: del archive[dname_meta]
        
        # Generate Historical Labels before loading particles
        id_ar, label_ar = tng_particles.get_hist_labels_dict( 
            t_df, t_snap, 
            save_file=True )        
        hist_df = pd.DataFrame( {'particle_id':id_ar, 'hist_label':label_ar} )
    
        # Get Raw Particles
        if dev_print: print("Requesting Particles")        
        t_pts, t_meta = tng_particles.std_merger_pts( p_id, s_id, t_snap )

        # Add new historical lables to meta
        t_meta = pd.merge( t_meta, hist_df, on='particle_id', how='left' )

        if dev_print:
            print(f"Stardardized Pts: {t_pts.shape} - Meta: {t_meta.shape}")
            print( '\n###  Particle Sample  ###', t_pts[:2], '\n' )
            print("###  Meta Sample  ###")
            print(t_meta.columns)
            print(t_meta.dtypes)
            print(t_meta.head())
            print(t_meta.tail())
            
        # Modify Particles into Final Reference Frame
        t_pts[:,:3]  = tf.apply_transformation( pos_mat, t_pts[:,:3]  )
        t_pts[:,3:6] = tf.apply_transformation( vel_mat, t_pts[:,3:6] )
        
        # Add Particles to Archive
        t_pts = np.asarray( t_pts, dtype=np.float32) # Reduce size
        a_pts = archive.create_dataset("particles_orf", data=t_pts)
        a_pts.attrs['headers'] =  np.array(pt_headers, dtype=h5py.string_dtype("utf-8"))
        a_pts.attrs['units']   =  np.array(pt_units,   dtype=h5py.string_dtype("utf-8"))

        # Add select Meta data
        meta_arr = t_meta[["particle_type", "subhalo_label", "hist_label"]].to_numpy()
        meta_arr = np.asarray(meta_arr, dtype=np.int8)        
        a_meta = archive.create_dataset("particles_meta", data=meta_arr )
        a_meta.attrs['headers'] = np.array(meta_headers, dtype=h5py.string_dtype("utf-8"))
        a_meta.attrs['particle_type_key'] = json.dumps(pt_type_key)

if buildEnv:
    
    create_target_archive( t_row, m4_df, dev_print=True, overwrite=True )

    print( 'done')


# In[ ]:


def read_target_archive(p_id=None, archive_loc=None, data_dir="tng-data", load_arrays=True):
    """
    Read a target-archive-<p_id>.h5 and return a user-friendly dict.

    Params
    ------
    p_id : int | str | None
        Primary subhalo id used in the filename. Provide either p_id or archive_loc.
    archive_loc : str | None
        Full path to the archive file. Provide either archive_loc or p_id.
    data_dir : str
        Used only if archive_loc is not provided.
    load_arrays : bool
        If True, loads datasets into numpy arrays. If False, returns shapes/dtypes only.

    Returns
    -------
    out : dict
        {
          "file": {"path": ..., "keys": [...]},
          "particles": {"data": ndarray or None, "shape":..., "dtype":..., "headers":[...], "units":[...]},
          "meta": {"data": ndarray or None, "shape":..., "dtype":..., "headers":[...], "particle_type_key": {...}},
        }
    """
    if archive_loc is None:
        if p_id is None:
            raise ValueError("Provide either p_id or archive_loc.")
        archive_loc = os.path.abspath(os.path.join(data_dir, f"target-archive-{p_id}.h5"))

    out = {
        "file": {"path": archive_loc, "keys": []},
        "particles": None,
        "meta": None,
    }

    with h5py.File(archive_loc, "r") as f:
        out["file"]["keys"] = list(f.keys())

        # ---- particles ----
        if "particles_orf" in f:
            d = f["particles_orf"]
            particles = {
                "shape": tuple(d.shape),
                "dtype": str(d.dtype),
                "headers": d.attrs.get("headers", None),
                "units": d.attrs.get("units", None),
            }

            # attrs might come back as numpy bytes; normalize to list[str]
            if particles["headers"] is not None:
                particles["headers"] = np.array(particles["headers"]).astype(str).tolist()
            if particles["units"] is not None:
                particles["units"] = np.array(particles["units"]).astype(str).tolist()

            particles["data"] = d[:] if load_arrays else None
            out["particles"] = particles

        # ---- meta ----
        if "particles_meta" in f:
            d = f["particles_meta"]
            meta = {
                "shape": tuple(d.shape),
                "dtype": str(d.dtype),
                "headers": d.attrs.get("headers", None),
                "particle_type_key": d.attrs.get("particle_type_key", None),
            }

            if meta["headers"] is not None:
                meta["headers"] = np.array(meta["headers"]).astype(str).tolist()

            # You stored JSON in an attribute; decode it back to dict
            if meta["particle_type_key"] is not None:
                val = meta["particle_type_key"]
                # h5py may return bytes or str
                if isinstance(val, (bytes, np.bytes_)):
                    val = val.decode("utf-8")

                # your writer currently does json.dumps(pt_type_key) where pt_type_key is *already* json
                # so handle both single-encoded and double-encoded cases safely
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, str):
                        parsed = json.loads(parsed)
                    meta["particle_type_key"] = parsed
                except Exception:
                    meta["particle_type_key"] = val  # fall back to raw

            meta["data"] = d[:] if load_arrays else None
            out["meta"] = meta

    return out


if buildEnv:
    # Example:
    arc = read_target_archive(p_id=t_row["p_SubhaloIDRaw"], data_dir="tng-data")
    pts = arc["particles"]["data"]
    meta = arc["meta"]["data"]
    print( f"Pts Shape:  {pts.shape}")
    print( f"Pts Header: {arc['particles']['headers']}")
    print( f"Pts Units:  {arc['particles']['units']}")
    print('')
    print( f"Meta Shape:  {meta.shape}")
    print( f"Meta Headers:  {arc['meta']['headers']}")
    print( f"Particle Type:  {arc['meta']['particle_type_key']}")
    
    tf.xy_histogram( pts, std_dev=1.5 )


# In[ ]:


def func_gen_target_archive( script_args ):
        
    # Read potential tng targets file
    all_loc = script_args.inputFile
    assert os.path.exists( all_loc ), f'Target information file not found: {all_loc}'
    print(f"Reading Target Dynamics: {all_loc}")    
    all_df = pd.read_pickle( all_loc )
    all_df['p_SubhaloIDRaw'] = all_df['p_SubhaloIDRaw'].astype(np.uint64)
    all_df['s_SubhaloIDRaw'] = all_df['s_SubhaloIDRaw'].astype(np.uint64)
    print( f"M4-DF: {all_df.shape}")
    
    t_id = script_args.targetID
    print( f"Getting Target Info: {t_id}")
    t_row = all_df[ all_df['p_SubhaloIDRaw'] == t_id ]
    if len( t_row ) != 1:
        raise ValueError("BAD" )
    t_row = t_row.iloc[0]
    
    overwrite = script_args.overwrite
    print( f"Overwrite?: {overwrite}")    
    
    create_target_archive( t_row, all_df, dev_print=False, overwrite=overwrite )

#     print( 'done')
    

if buildEnv and True:  # Use compiled script from cmd line for speed. 
    
    # Command Line Arguments
    cmdStr  = 'python3 tng_archive_targets.py'
    cmdStr += ' --function gen_target_archive'
    cmdStr += ' --dataDir tng-data/'
    cmdStr += ' --inputFile tng-targets/moi-4-targets.pkl'
    # cmdStr += ' --targetID 62000000190717'
    cmdStr += ' --targetID 75000000306160'
    # cmdStr += ' --overwrite'
    
    # Read string as if command line
    print( "CMD Line: \n\t$", cmdStr)
    
    # This function doesn't like the 'python3 my-script.py' part.
    script_args = parser.parse_args(cmdStr.split()[2:])
    
if script_args.function == 'gen_target_archive':
    func_gen_target_archive( script_args )
    print('DONE')


# In[5]:


if buildEnv:
    import os, subprocess, sys

    cmd = f"{sys.executable} tng_archive_targets.py --function gen_target_archive --dataDir tng-data/ --inputFile tng-targets/moi-4-targets.pkl --targetID %s"

    tid_list = []

    with open( "tng-targets/moi-4-top-ratings.txt" ) as f:
        for l in f:
            tid_list.append( l.strip() )

    print( tid_list )

    for tid in tid_list:
        t_cmd = cmd%tid
        print( f"Running CMD: {t_cmd}")


# In[ ]:


if buildEnv:
    import os, subprocess, sys

    base_cmd = [
        sys.executable, "tng_archive_targets.py",
        "--function", "gen_target_archive",
        "--dataDir", "tng-data/",
        "--inputFile", "tng-targets/moi-4-targets.pkl",
        "--targetID",
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")

    for tid in tid_list:
        full_cmd = base_cmd + [tid]
        print("Running CMD:", " ".join(full_cmd))

        r = subprocess.run(full_cmd, text=True, capture_output=True, env=env)

        if r.returncode != 0:
            print(f"[WARN] targetID {tid} failed (rc={r.returncode})")
            print(r.stderr)
        else:
            print(f"[OK] targetID {tid} completed")


# In[ ]:





# In[ ]:




