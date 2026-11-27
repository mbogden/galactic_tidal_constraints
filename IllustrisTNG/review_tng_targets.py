#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: review_tng_target_images.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2024-Apr-11

Description:

References:  Sections of this code were written with the assistance of ChatGPT made by OpenAI.

"""
# ================================ IMPORTS ================================ #

import math, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

import galaxyJSPAM.IllustrisTNG.tng_functions as tf
# import galaxyJSPAM.IllustrisTNG.convert_tng_to_spam as cts
# !pip install illustris-python
# import IllustrisTNG.tng_images as ti


# Set NumPy print options to reduce the precision to 3 decimal places
np.set_printoptions(precision=2)

print("Import Done")


# In[6]:


#Let's load some target data

all_targets_df = pd.read_pickle('/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-3-dynamics-dict-orbital-frame.pkl')
print( all_targets_df.shape)


# In[7]:


def get_all_targets_meta(hdf_file):
    """
    Retrieves metadata and image names for all targets found in the HDF5 file.

    Args:
    -----
    hdf_file (str): Path to the HDF5 file.

    Returns:
    --------
    all_targets_meta (dict): A dictionary where the keys are target IDs (pids) and the values are dictionaries
                             containing the target metadata and a list of image names.
    """
    all_targets_meta = {}
    
    if not os.path.exists( hdf_file ):
        return {}

    with h5py.File(hdf_file, 'r') as hdf:
        # Iterate through all groups in the HDF5 file
        for group_name in hdf.keys():
            if group_name.startswith('Target_'):
                # Extract the target ID (pid) from the group name
                pid = group_name

                # Access the target group
                target_group = hdf[group_name]

                # Retrieve the target metadata
                t_meta = {key: target_group.attrs[key] for key in target_group.attrs}

                # Get the list of image names (keys starting with 'Image_')
                image_names = [key for key in target_group.keys() if key.startswith('Image_')]

                # Store metadata and image names in the dictionary with pid as the key
                all_targets_meta[pid] = {
                    'metadata': t_meta,
                    'target_name': pid,
                    'image_names': image_names
                }

    return all_targets_meta

if False:
    test_img_file = '/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-images/moi-3-dynamics-dict-std-images.h5'
    all_meta = get_all_targets_meta( test_img_file )
    print('Target Images:', len( all_meta.keys() ))


# In[8]:


if False:
    print( list(all_meta.keys())[0] )
    print( type( list(all_meta.keys())[0] ) )

    sr_list = []
    for i, pid in enumerate(all_meta.keys()):
        print( i, pid, end='\r' )
        t_row = pd.Series( all_meta[pid]['metadata'] )
        t_row['target_name'] = all_meta[pid]['target_name']
        t_row['img_names'] = all_meta[pid]['image_names']
        sr_list.append( t_row )

    img_df = pd.DataFrame( sr_list )
    img_df = img_df.sort_values(by=['moi_2','snap'], ascending=[True, True])

    print( img_df.shape )
    print( img_df.columns )



# In[9]:


if False:

    pid_list = img_df['moi_2'].unique()
    t_row = img_df[img_df['moi_2'] == pid_list[3]].iloc[-3]

    print( t_row.shape )

    t_df = img_df[img_df['moi_2'] == t_row['moi_2']]

    print( t_df.shape )

    tp_path = np.stack(t_df['pa_SubhaloPos'].values)
    ts_path = np.stack(t_df['sa_SubhaloPos'].values)

    def plot_path( tp_path, ts_path, title='' ):
        plt.plot( tp_path[:,0], tp_path[:,1], label='p')
        plt.plot( ts_path[:,0], ts_path[:,1], label='s')
        plt.legend()
        plt.title( title )
        plt.show()
        
    plot_path( tp_path, ts_path, title='Raw Pts' )


    for i in range( tp_path.shape[0] ):
        tp_path[i] = tf.pt_apply_transformation( tp_path[i], t_df.iloc[i]['mod1_pos_transform'] )
        ts_path[i] = tf.pt_apply_transformation( ts_path[i], t_df.iloc[i]['mod1_pos_transform'] )

    plot_path( tp_path, ts_path, 'Mod 1' )
        
    for i in range( tp_path.shape[0] ):
        tp_path[i] = tf.pt_apply_transformation( tp_path[i], t_row['mod2_pos_transform'] )
        ts_path[i] = tf.pt_apply_transformation( ts_path[i], t_row['mod2_pos_transform'] )

    plot_path( tp_path, ts_path, 'Mod 2' )
        
    for i in range( tp_path.shape[0] ):
        tp_path[i] = tf.pt_apply_transformation( tp_path[i], t_row['mod3_pos_transform'] )
        ts_path[i] = tf.pt_apply_transformation( ts_path[i], t_row['mod3_pos_transform'] )

    plot_path( tp_path, ts_path, title='Mod 3' )
        
    for i in range( tp_path.shape[0] ):
        tp_path[i] = tf.pt_apply_transformation( tp_path[i], t_row['mod4_pos_transform'] )
        ts_path[i] = tf.pt_apply_transformation( ts_path[i], t_row['mod4_pos_transform'] )

    plot_path( tp_path, ts_path, title='Mod 4' )
    
    # tp_path = tf.pt_apply_transformation( tp_path, t_row['orbital_frame_pos_transform'] )
    # ts_path = tf.pt_apply_transformation( ts_path, t_row['orbital_frame_pos_transform'] )



# In[10]:


def read_img_data(img_file, pid, img_name_list):
    """
    Reads the image data and metadata from the HDF5 file.

    Args:
    -----
    img_file (str): Path to the HDF5 file.
    pid (str): Target ID.
    img_name_list (list): List of image names to read.

    Returns:
    --------
    img_data (dict): A dictionary where the keys are image names and the values are dictionaries containing
                     the image data ('data') and the associated metadata ('metadata').
    """
    img_data = {}
    with h5py.File(img_file, 'r') as hdf:
        for img_name in img_name_list:
            path_name = f'{pid}/{img_name}'

            # Check if the path corresponds to a dataset or a group
            if isinstance(hdf[path_name], h5py.Dataset):
                # Retrieve image data and metadata (if it's directly a dataset)
                image_dataset = hdf[path_name]
                img_data[img_name] = {
                    'data': image_dataset[()],  # Image data
                    'metadata': {key: image_dataset.attrs[key] for key in image_dataset.attrs}  # Metadata
                }

            elif isinstance(hdf[path_name], h5py.Group):
                # If it's a group, look inside for datasets (assuming 'histogram_matrix' is the dataset)
                dataset_name = list(hdf[path_name].keys())[0]  # Assuming the first dataset
                image_dataset = hdf[f'{path_name}/{dataset_name}']

                # Retrieve image data and metadata
                img_data[img_name] = {
                    'data': image_dataset[()],  # Image data
                    'metadata': {key: image_dataset.attrs[key] for key in image_dataset.attrs}  # Metadata
                }

            else:
                print(f"Warning: {path_name} is not a valid dataset or group.")

    return img_data

if False:
    print( 'Img Names:', type(t_row['img_names']),  t_row['img_names'] )
    print( 'Target Name:', type( t_row['target_name'] ), t_row['target_name'] )

    img_data = read_img_data( test_img_file, t_row['target_name'], t_row['img_names'] )

    print( 'Img Data:', type(img_data), len(img_data.keys()) )
    print( 'Img Data Keys:', img_data.keys() )
    print( 'First Img:', img_data[t_row['img_names'][0]]['data'].shape )
    print("Meta names:", img_data[t_row['img_names'][0]]['metadata'].keys() )

    # plt.imshow( img_data[t_row['img_names'][0]] )


# In[ ]:


def visualize_image(img_in, ax_in = None, pixel_centers=None, log_scale=True, cmap='gray', title='', fig_size=None):
    """
    Visualizes the histogram matrix with optional log scaling.

    Args:
    -----
    img_in (np.ndarray): The histogram matrix to display.
    ax_in (matplotlib.axes.Axes): The axis to plot the image on.
    pixel_centers (list): List of pixel centers to plot on the image.
    log_scale (bool): Whether to apply logarithmic scaling for visualization.
    cmap (str): Colormap for rendering the image.

    Returns:
    --------
    None
    """
    
    img = img_in.copy()
    # Apply log scaling if desired
    if log_scale:
        # Avoid taking log of zero
        img = np.log1p(img)
    
    # Normalize the data to [0, 1]
    h_min = np.min(img)
    h_max = np.max(img)
    img_normalized = (img - h_min) / (h_max - h_min) 

    if ax_in is None:
        if fig_size is not None:
            plt.figure(figsize=fig_size)
        else:
            plt.figure()
        ax = plt.gca()
    else:
        ax = ax_in

    
    # If pixel_center is provided, plot it on the image
    if pixel_centers is not None:
        for center in pixel_centers:
            # plt.scatter(center[1], center[0], c='white', s=50, edgecolors='black', label='Center')
            ax.scatter(center[1], center[0], c='white', s=50, edgecolors='black', label='Center')
    
    # plt.imshow(img_normalized.T, cmap=cmap, origin='lower')
    ax.imshow(img_normalized.T, cmap=cmap, origin='lower')
 
    # plt.axis('off')
    ax.axis('off')
    ax.set_title(title)
   

    # plt.show()
if False:
    print( img_data[t_row['img_names'][0]]['data'].shape )
    visualize_image( img_data[t_row['img_names'][0]]['data'], 
                    title=img_data[t_row['img_names'][0]]['metadata']['name'],
                    fig_size=(8, 8) )   


# In[22]:


# Plot the orbit and all the images together
def plot_orbit_and_images(tp_path, ts_path, img_data, img_names, \
                          title='', fig_size = 8, save_path=None,\
                            show_plot=True):
    """
    Plots the orbit path and the images side by side.

    Args:
    -----
    tp_path (np.ndarray): The path of the target galaxy.
    ts_path (np.ndarray): The path of the satellite galaxy.
    img_data (dict): A dictionary containing the image data.
    img_names (list): A list of image names to plot.
    title (str): Title for the plot.

    Returns:
    --------
    None
    """
    num_images = len(img_names)

    # Plot vertically
    # fig, axs = plt.subplots(1, num_images + 1, figsize=(5 * (num_images + 1), 5))
    # fig, axs = plt.subplots( num_images + 1, 1, figsize=(fig_size, fig_size * (num_images + 1)))

    # Should always be a 2x3 grid

    fig, axs = plt.subplots(2, 3, figsize=(fig_size * 3, fig_size * 2))

    plt.title(title)

    #['Image_orbital_frame', 
    # 'Image_tidal_primary_all', 'Image_tidal_primary_solo', 
    # 'Image_tidal_secondary_all', 'Image_tidal_secondary_solo']

    # Plot the orbit path
    axs[0,0].plot(tp_path[:, 0], tp_path[:, 1], label='Primary')
    axs[0,0].plot(ts_path[:, 0], ts_path[:, 1], label='Secondary')
    axs[0,0].set_title('Orbit Path')

    # Turn labels off
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])

    # Set square aspect ratio
    # axs[0,0].set_aspect('equal', adjustable = 'datalim')
    axs[0,0].legend()

    name = 'Image_orbital_frame'
    visualize_image( img_data[name]['data'], 
                    ax_in=axs[1,0], 
                    title=name )

    name = 'Image_tidal_primary_solo'
    visualize_image( img_data[name]['data'], 
                    ax_in=axs[0,1], 
                    title=name )

    name = 'Image_tidal_primary_all'
    visualize_image( img_data[name]['data'], 
                    ax_in=axs[1,1], 
                    title=name )

    name = 'Image_tidal_secondary_solo'
    visualize_image( img_data[name]['data'], 
                    ax_in=axs[0,2], 
                    title=name )

    name = 'Image_tidal_secondary_all'
    visualize_image( img_data[name]['data'], 
                    ax_in=axs[1,2], 
                    title=name )



    # # Plot the images
    # for i, img_name in enumerate(img_names):
    #     img = img_data[img_name]['data']
    #     img_title = img_data[img_name]['metadata']['name']
    #     visualize_image(img, ax_in=axs[i+1], title=img_title,)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        
    if show_plot:
        plt.show()
if False:

    plot_orbit_and_images(tp_path, ts_path, img_data, t_row['img_names'], 
                        title=f'Orbit Path and Images\n{pid}', fig_size=7, 
                        save_path=f'test_plot_{pid}.png')

    plot_orbit_and_images(tp_path, ts_path, img_data, t_row['img_names'], 
                        title=f'Orbit Path and Images\n{pid}', fig_size=7, 
                        save_path=f'test_plot_{pid}.png', show_plot=False)


# In[13]:


def gen_img_df_and_path( img_loc ):

    print("Reading HDF5 Image File: ", img_loc)
    all_meta = get_all_targets_meta( img_loc )
    print('Target Images:', len( all_meta.keys() ))

    # Create a DataFrame with the metadata
    sr_list = []
    for i, pid in enumerate(all_meta.keys()):
        print( i, pid, end='\r' )
        t_row = pd.Series( all_meta[pid]['metadata'] )
        t_row['target_name'] = pid
        t_row['img_names'] = all_meta[pid]['image_names']
        sr_list.append( t_row )

    print('\n')
    raw_img_df = pd.DataFrame( sr_list )
    raw_img_df = raw_img_df.sort_values(by=['moi_2','snap'], ascending=[True, True])

    print( 'Image DF:', raw_img_df.shape )

    new_rows = []
    for i, t_row in raw_img_df.iterrows():

        t_df = raw_img_df[raw_img_df['moi_2'] == t_row['moi_2']]
        tp_path = np.stack(t_df['pa_SubhaloPos'].values)
        ts_path = np.stack(t_df['sa_SubhaloPos'].values)

        for i in range( tp_path.shape[0] ):
            tp_path[i] = tf.pt_apply_transformation( tp_path[i], t_df.iloc[i]['mod1_pos_transform'] )
            ts_path[i] = tf.pt_apply_transformation( ts_path[i], t_df.iloc[i]['mod1_pos_transform'] )

        for i in range( tp_path.shape[0] ):
            tp_path[i] = tf.pt_apply_transformation( tp_path[i], t_row['mod2_pos_transform'] )
            ts_path[i] = tf.pt_apply_transformation( ts_path[i], t_row['mod2_pos_transform'] )

        for i in range( tp_path.shape[0] ):
            tp_path[i] = tf.pt_apply_transformation( tp_path[i], t_row['mod3_pos_transform'] )
            ts_path[i] = tf.pt_apply_transformation( ts_path[i], t_row['mod3_pos_transform'] )

        for i in range( tp_path.shape[0] ):
            tp_path[i] = tf.pt_apply_transformation( tp_path[i], t_row['mod4_pos_transform'] )
            ts_path[i] = tf.pt_apply_transformation( ts_path[i], t_row['mod4_pos_transform'] )

        new_row = t_row.copy()
        new_row['tp_path'] = tp_path
        new_row['ts_path'] = ts_path
        new_rows.append( new_row )
    
    img_df = pd.DataFrame( new_rows )

    # Read image data for each target and save as png
    c = 0
    for i, t_row in img_df.iterrows():
        print( i, t_row['target_name'], end='\r' )
        img_data = read_img_data( img_loc, t_row['target_name'], t_row['img_names'] )

        img_path = f'/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-images/moi_4_search/{t_row["target_name"]}.png'


        plot_orbit_and_images(t_row['tp_path'], t_row['ts_path'], img_data, t_row['img_names'], 
                      title=f'Orbit Path and Images\n{t_row["target_name"]}', fig_size=8, 
                      save_path=img_path)
        c += 1
        # if c > 5:
        #     break





    
if False:

    gen_img_df_and_path( test_img_file )


# In[15]:


def get_target_images( img_file, pid, img_names ):
    """
    Retrieves target images from the HDF5 file.

    Args:
    -----
    img_file (str): Path to the HDF5 file.
    pid (str): Target ID.
    img_names (list): List of image names.

    Returns:
    --------
    images (dict): A dictionary where the keys are image names and the values are the corresponding images.
    """
    images = {}

    with h5py.File(img_file, 'r') as hdf:
        target_group = hdf[pid]

        for img_name in img_names:
            images[img_name] = target_group[img_name][()]

    return images

if False:
    pid = path_df['moi_2'].unique()[0]

    t_row = path_df[path_df['moi_2'] == pid].iloc[0]

    img_dict = get_target_images( test_img_file, t_row['img_target_name'], t_row['img_names'] )

    print( img_dict.keys() )

def plot_images( img_dict, n_cols=2 ):
    """
    Plots the images in the dictionary.

    Args:
    -----
    img_dict (dict): A dictionary where the keys are image names and the values are the corresponding images.
    """
    n_images = len(img_dict)

    n_rows = int(math.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for i, (img_name, img) in enumerate(img_dict.items()):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(img, cmap='gray')
        ax.set_title(img_name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ---
# ---
# ---
# # Rating Images
# 
# At this point I used a seperate script named "rate_tng_target_images.py" to review all 1200+ target images and gave them a 0-5 rating depending on how promenent the tidal features were for the primary and secondary galaxies.  It created a csv file I will now load and review. 

# In[16]:


if True:
    m3_df = pd.read_pickle('/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-3-dynamics-dict-orbital-frame.pkl')
    print( m3_df.shape )

    # Assume csv file has no headers
    m4_ratings_raw = pd.read_csv('/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-4-ratings.csv', header=None)
    print( m4_ratings_raw.shape )


# In[17]:


if True:

    # Loop through the rows of the ratings and expand on data
    new_rows = []

    for i, row in m4_ratings_raw.iterrows():
        # print( row[0], row[1], end='\n' )

        new_row = {}
        new_row['img_name_1'] = row[0]
        new_row['moi_4_rating'] = row[1]
        p_SubhaloIDRaw = int( row[0].split('_')[1].split('.')[0])
        new_row['p_SubhaloIDRaw'] = p_SubhaloIDRaw

        new_rows.append( new_row )

        print( i, m4_ratings_raw.shape, end='\r' )

    m4_ratings = pd.DataFrame( new_rows )


# In[18]:


if True:

    # Search for duplicate rows with the same SubhaloIDRaw

    # Use the row with the highest rating

    m4_ratings = m4_ratings.sort_values(by=['moi_4_rating'], ascending=False)
    m4_ratings = m4_ratings.drop_duplicates(subset='p_SubhaloIDRaw', keep='first')

    print( m4_ratings.shape )


# In[19]:


if True:
    # Merge with m3_df on p_SubhaloIDRaw

    m4_df = m3_df.merge(m4_ratings, left_on='p_SubhaloIDRaw', right_on='p_SubhaloIDRaw', how='inner')
    print( 'M4_DF: ', m4_df.shape )


# In[24]:


if True:

    def plot_ratings( ratings, title = 'Ratings', legend_loc = 'upper right' ):

        # Remove/clear any existing plot
        plt.clf()

        # Create histogram
        plt.figure(figsize=(8, 5))
        counts, bins, bars = plt.hist(ratings, bins=np.arange(0, 7) - 0.5, edgecolor='black', rwidth=0.8)

        # Label each bar with its count
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, count, int(count), ha='center', va='bottom')

        # Center x-ticks under each bin
        plt.xticks(range(6), range(6))

        # Labels and title
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.title(title)

        # Manual Legend description at top and right corner
        if legend_loc == 'upper right':
            plt.text( 0.9, 0.8, 'Tidal Features\n0-1: None\n2: Slight\n3: Some\n4: Good\n5: Great', 
                horizontalalignment='center', verticalalignment='center',
               transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.5))
        elif legend_loc == 'upper left':
            plt.text( 0.1, 0.8, 'Tidal Features\n0-1: None\n2: Slight\n3: Some\n4: Good\n5: Great', 
                horizontalalignment='center', verticalalignment='center',
               transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.5))

        # Display plot
        plt.show()

    # histogram all ratings
    plot_ratings( m4_ratings['moi_4_rating'], f'All Ratings\nTotal: {m4_ratings.shape[0]}' )



# In[25]:


if True:
    
    # Get unique values in 'moi_2' field
    moi_2_list = m4_df['moi_2'].unique()
    m2_ratings = []

    # Loop through the unique values and get all rows with that value
    for moi_2 in moi_2_list:
        t_df = m4_df[m4_df['moi_2'] == moi_2]

        # Get the value of the highest rating in moi_4_rating
        max_rating = t_df['moi_4_rating'].max()
        m2_ratings.append( [ moi_2, max_rating ] )

    m2_ratings = np.array(m2_ratings)
    print ( 'Moi_2 List:', m2_ratings.shape )

    # Histogram of the max ratings
    plot_ratings( m2_ratings[:,1], f'Max Rating per Collision\nTotal: {len(m2_ratings)}', legend_loc='upper left' )



# In[31]:


def plot_random_rating_grid(m4_df, img_dir, n_samples=4, rating_col='moi_4_rating',
                            figsize_per_panel=3.0, random_state=None,
                            crop_bottom_left_frac=(0.45, 0.3)):
    """
    For each rating, grab up to n_samples random images and plot them in a grid:
    columns = ratings (low -> high), rows = examples.

    If crop_bottom_left_frac is not None, only show a fixed bottom-left region
    of each composite image.

    Parameters
    ----------
    m4_df : pandas.DataFrame
        DataFrame containing at least [rating_col, image-name column].
    img_dir : str
        Directory containing the PNG images.
    n_samples : int
        Number of random images per rating (max).
    rating_col : str
        Name of the column with rating values (default: 'moi_4_rating').
    figsize_per_panel : float
        Size multiplier per subplot (in inches).
    random_state : int or None
        Optional random seed for reproducibility.
    crop_bottom_left_frac : tuple or None
        (h_frac, w_frac). If given, crop to the bottom h_frac of height and
        left w_frac of width, e.g. (0.5, 0.5) = bottom-left quarter.
    """

    # Choose which column has the image filenames
    if 'img_name' in m4_df.columns:
        img_col = 'img_name'
    elif 'img_name_1' in m4_df.columns:
        img_col = 'img_name_1'
    else:
        raise ValueError("Could not find 'img_name' or 'img_name_1' in m4_df columns.")

    # Sorted unique ratings, low -> high
    ratings = np.sort(m4_df[rating_col].unique())
    n_ratings = len(ratings)

    # Grid: rows = examples, cols = ratings
    n_rows = n_samples
    fig, axes = plt.subplots(
        n_rows, n_ratings,
        figsize=(figsize_per_panel * n_ratings, figsize_per_panel * n_rows)
    )

    # Ensure axes is 2D
    if n_rows == 1 and n_ratings == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_ratings == 1:
        axes = axes[:, np.newaxis]

    rng = np.random.default_rng(random_state)

    for col_idx, rating in enumerate(ratings):
        sub_df = m4_df[m4_df[rating_col] == rating]
        if sub_df.empty:
            for row_idx in range(n_rows):
                axes[row_idx, col_idx].axis('off')
            continue

        k = min(n_samples, len(sub_df))
        sample_indices = rng.choice(sub_df.index, size=k, replace=False)
        samples = sub_df.loc[sample_indices]

        axes[0, col_idx].set_title(f'Rating {rating}')

        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            ax.axis('off')

            if row_idx >= k:
                continue

            img_name = samples.iloc[row_idx][img_col]
            img_path = os.path.join(img_dir, img_name)

            try:
                img = plt.imread(img_path)

                # --- CROP BOTTOM-LEFT IF REQUESTED ---
                if crop_bottom_left_frac is not None:
                    h_frac, w_frac = crop_bottom_left_frac
                    h, w = img.shape[:2]
                    ch = int(h * h_frac)
                    cw = int(w * w_frac)
                    # bottom h_frac of height, left w_frac of width
                    img = img[h - ch:h, 0:cw]
                # --------------------------------------

                ax.imshow(img)
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'Missing\nimage',
                        ha='center', va='center', transform=ax.transAxes)

    fig.suptitle(f'Random Examples per Rating (n={n_samples})')
    plt.tight_layout()
    plt.show()


if True:
    img_dir = '/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-images/moi_4_search/'
    plot_random_rating_grid(m4_df, img_dir, n_samples=3, random_state=42)


# In[ ]:


# Save the m4_df to a pickle file

if True:

    m4_df.to_pickle('/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-4-targets.pkl')

    # and csv for a human readable version
    m4_df.to_csv('/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-4-targets.csv')


# In[ ]:


if True:

    # Let's filter the top ratings (4 or 5) and save those p_SubhaloIDRaw to a txt file

    top_ratings = m4_df[m4_df['moi_4_rating'] >= 4]
    p_id_list = top_ratings['p_SubhaloIDRaw'].values

    with open('/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-targets/moi-4-top-ratings.txt', 'w') as f:
        for p_id in p_id_list:
            f.write(f'{int(p_id)}\n')


# In[33]:


if True:

    # Loop through ALL rows with 5 or 4, print the p_SubhaloIDRaw and plot the image
    img_dir = '/home/mbo2d/galStuff/galaxyJSPAM/IllustrisTNG/tng-images/moi_4_search/'

    for rating in [5, 4]:
        tmp_df = m4_df[m4_df['moi_4_rating'] == rating]
        print( rating, tmp_df.shape )

        for i, row in tmp_df.iterrows():
            print( row['p_SubhaloIDRaw'] )

            img_path = img_dir + row['img_name_1']
            img = plt.imread( img_path )

            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Rating: {rating}\nSubhaloIDRaw: {row["p_SubhaloIDRaw"]}')
            plt.show()

            # break


# In[ ]:




