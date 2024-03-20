# exp_duration_investigation.py
# Code to investigate duration of model training

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 8th September 2023

# Import required modules
from pathlib import Path
import pandas as pd

# Specify dataset
dset = 'Lightbox'

# Select relevant dictionary
if dset == 'Sunlamp':
    
    # List of paths to files with epoch duration info
    filepath_list = [Path('work_dirs/litehrnet_30_speedplus_640x640_slp_new/20230906_121820.log'),
                     Path('work_dirs/litehrnet_30_speedplus_640x640_slp_new_aug/20230906_195216.log')]
    
    # Key epoch indicator
    key_ind = '[50/419]'

elif dset == 'Lightbox':

    # List of paths to files with epoch duration info
    filepath_list = [Path('work_dirs/litehrnet_30_speedplus_640x640_lbx_new/20230907_210150.log'),
                     Path('work_dirs/litehrnet_30_speedplus_640x640_lbx_new_aug/20230907_031529.log')]
    
    # Key epoch indicator
    key_ind = '[50/1011]'

# Preallocate for median time taken per 50 epochs
time_list = []

# For each file
for filepath in filepath_list:

    # Open file
    with open(filepath, 'r') as f:
        
        # Create a list of all lines
        all_lines = f.readlines()

        # Preallocate string for relevant timing info
        timing_info = []

        # Find lines with relevant timing information
        for ind_line in all_lines:

            # Print line
            print(ind_line)

            # If line contains relevant information, update timing info
            if key_ind in ind_line:
                timing_info.append(ind_line[:19])

    # Create pandas DataFrame of times
    df = pd.to_datetime(timing_info)

    # Time taken per epoch
    time_per_epoch = df[1:] - df[:-1]

    # Update time_list
    time_list.append(time_per_epoch.median())                                                                          

# Print median times per 50 epochs
print(f'No aug: {time_list[0]}, geo aug: {time_list[1]}')

