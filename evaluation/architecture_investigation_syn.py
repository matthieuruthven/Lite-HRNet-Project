# architecture_investigation_syn.py
# Code to create percentage of correct keypoint (PCK) plots
# to enable investigation into effect of HRNet architecture
# on Lite-HRNet accuracy

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 25th September 2023

# Import required modules
from pathlib import Path
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# Path to folder containing all results folders
root_dir_path = Path('/data/mruthven/lite-hrnet/work_dirs')

# List of names of folders containing JSON files of results
res_dir_list = [
    'litehrnet_18_speedplus_640x640_syn_lbx_aug_pretrain_mpii',
    'litehrnet_18_speedplus_640x640_syn_slp_aug_pretrain_mpii',
    'litehrnet_30_speedplus_640x640_syn_lbx_aug_pretrain_mpii',
    'litehrnet_30_speedplus_640x640_syn_slp_aug_pretrain_mpii',
    ]

# Preallocate pandas DataFrame for results
all_df = pd.DataFrame()

# Number of synthetic images
n_syn = 47966

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

    # JSON file name
    file_name = os.listdir(root_dir_path / res_dir)
    file_name = [idx for idx in file_name if idx.endswith('.json')]
    assert len(file_name) == 1, f'There are multiple JSON files in {res_dir}'
    file_name = file_name[0]
    
    # Read JSON file using pandas
    df = pd.read_json(root_dir_path / res_dir / file_name, lines=True)

    # Extract results on validation dataset
    df = df[df['mode'] == 'val']

    # Extract relevant columns
    df = df[['epoch', 'PCKh']]

    # Add columns
    df['Architecture'] = f'Lite-HRNet-{arch}'
    df['Dataset'] = dset
    df['Augmentation'] = 'Geometric'
    df['Pretraining'] = 'MPII'

    # Update all_df
    all_df = pd.concat([all_df, df])

# Rename columns
all_df.rename(columns={'epoch': 'Epoch', 'PCKh': 'Mean Percentage Correct Keypoints'}, inplace=True)

# Set seaborn style
sns.set_style('darkgrid')

# Create seaborn plot
g = sns.relplot(data=all_df,
                kind='line',
                x='Epoch',
                y='Mean Percentage Correct Keypoints',
                hue='Architecture',
                style='Dataset',
                style_order=['Lightbox', 'Sunlamp'],
                errorbar=None)

# Save plot
plt.savefig(root_dir_path.parent / f'Architecture_Investigation_Val_Dset_PCK_SPEEDplus_{datetime.date.today()}_syn.png', bbox_inches='tight')

# Close plot
plt.close('all')