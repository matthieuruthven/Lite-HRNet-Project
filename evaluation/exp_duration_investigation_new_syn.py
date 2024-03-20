# exp_duration_investigation_new_syn.py
# Code to investigate duration of model training

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 25th September 2023

# Import required modules
from pathlib import Path
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Path to folder containing all results folders
root_dir_path = Path('/data/mruthven/lite-hrnet/work_dirs')

# List of names of folders containing LOG files of results
res_dir_list = [
    'litehrnet_18_speedplus_640x640_syn_lbx_aug_pretrain_mpii',
    'litehrnet_18_speedplus_640x640_syn_slp_aug_pretrain_mpii',
    'litehrnet_30_speedplus_640x640_syn_lbx_aug_pretrain_mpii',
    'litehrnet_30_speedplus_640x640_syn_slp_aug_pretrain_mpii',
    ]

# Preallocate pandas DataFrame for results
all_df = pd.DataFrame()

# Key epoch indicator
key_ind = '[50/11992]'

# Number of epochs
n_epochs = 100

# For each folder
for idx, res_dir in enumerate(res_dir_list):
                
    # Determine dataset
    if 'lbx' in res_dir:
        dset = 'Lightbox'
    elif 'slp' in res_dir:
        dset = 'Sunlamp'

    # Determine Lite-HRNet architecture
    arch = res_dir.split('_')
    arch = int(arch[1])

    # LOG file name
    file_name = os.listdir(root_dir_path / res_dir)
    file_name = [idx for idx in file_name if idx.endswith('.log')]
    assert len(file_name) == 1, f'There are multiple log files in {res_dir}'
    file_name = file_name[0]

    # Preallocate for median time taken per 50 epochs
    time_list = []

    # Open file
    with open(root_dir_path / res_dir / file_name, 'r') as f:
        
        # Create a list of all lines
        all_lines = f.readlines()

        # Preallocate string for relevant timing info
        timing_info = []

        # Find lines with relevant timing information
        for ind_line in all_lines:

            # If line contains relevant information, update timing info
            if key_ind in ind_line:
                timing_info.append(ind_line[:19])

                # Print line
                print(ind_line)

    # Create pandas DataFrame of times
    df = pd.to_datetime(timing_info)

    # Time taken per epoch
    time_per_epoch = df[1:] - df[:-1]
    time_per_epoch = time_per_epoch.median() * n_epochs

    # Add columns
    df = pd.DataFrame({'Architecture': [f'Lite-HRNet-{arch}'],
                       'Dataset': [dset],
                       'Augmentation': ['Geometric'],
                       'Pretraining': ['MPII'],
                       'Time Per Experiment (Days)': [round(time_per_epoch / pd.Timedelta(days=1), 2)]})

    # Update all_df
    all_df = pd.concat([all_df, df])

# Set seaborn style
sns.set_style('darkgrid')

# Create seaborn plot
g = sns.barplot(data=all_df,
                y='Dataset',
                x='Time Per Experiment (Days)',
                hue='Architecture',
                order=['Lightbox', 'Sunlamp'])

# Save plot
plt.savefig(root_dir_path.parent / 'Architecture_Investigation_Experiment_Duration_SPEEDplus_syn.png', bbox_inches='tight')

# Close plot
plt.close('all')