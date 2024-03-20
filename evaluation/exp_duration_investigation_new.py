# exp_duration_investigation_new.py
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

# List of paths (relative to root_dir_path) to JSON files of results
res_path_list = [
    Path(f'litehrnet_18_speedplus_640x640_lbx/20230921_090932.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_lbx_pretrain_mpii/20230922_024148.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_lbx_pretrain_coco/20230922_181924.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_lbx_aug/20230920_093218.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_lbx_aug_pretrain_mpii/20230922_233328.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_lbx_aug_pretrain_coco/20230922_124454.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_slp/20230920_204837.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_slp_pretrain_mpii/20230921_204552.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_slp_pretrain_coco/20230922_135025.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_slp_aug/20230921_013124.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_slp_aug_pretrain_mpii/20230921_141338.log.json'),
    Path(f'litehrnet_18_speedplus_640x640_slp_aug_pretrain_coco/20230922_081409.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_lbx_new/20230907_210150.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_lbx_new_pretrain_mpii/20230921_212758.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_lbx_new_pretrain_coco/20230922_221216.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_lbx_new_aug/20230907_031529.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_lbx_new_aug_pretrain_mpii/20230921_212428.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_lbx_new_aug_pretrain_coco/20230922_221138.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_slp_new/20230906_121820.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_slp_new_pretrain_mpii/20230921_135943.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_slp_new_pretrain_coco/20230922_150156.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_slp_new_aug/20230906_195216.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_slp_new_aug_pretrain_mpii/20230921_140500.log.json'),
    Path(f'litehrnet_30_speedplus_640x640_slp_new_aug_pretrain_coco/20230922_145959.log.json')
    ]

# Preallocate pandas DataFrame for results
all_df = pd.DataFrame()

# For each file
for res_path in res_path_list:
                
    # Folder name
    dir_name = str(res_path.parent)

    # File name
    file_name = res_path.name

    # List files in folder
    file_list = os.listdir(root_dir_path / dir_name)
    
    # List JSON files in folder
    file_list = [idx for idx in file_list if idx.endswith('.json')]

    # Check file
    assert len(file_list) == 1, 'There are multiple JSON files'
    assert file_name == file_list[0]

    # Determine dataset
    if 'lbx' in str(res_path.parent):
        dset = 'Lightbox'
    elif 'slp' in str(res_path.parent):
        dset = 'Sunlamp'
    
    # Determine if augmentation used
    if 'aug' in str(res_path.parent):
        aug_type = 'Geometric'
    else:
        aug_type = 'None'

    # Determine if pretrained model
    if 'mpii' in str(res_path.parent):
        pretraining = 'MPII'
    elif 'coco' in str(res_path.parent):
        pretraining = 'COCO'
    else:
        pretraining = 'None'

    # Determine Lite-HRNet architecture
    arch = str(res_path.parent).split('_')
    arch = int(arch[1])

    # Select relevant epoch indicator
    if dset == 'Sunlamp':
        
        # Key epoch indicator
        key_ind = '[50/419]'

    elif dset == 'Lightbox':

        # Key epoch indicator
        key_ind = '[50/1011]'

    # Preallocate for median time taken per 50 epochs
    time_list = []

    # Open file
    with open(root_dir_path / res_path.parent / res_path.stem, 'r') as f:
        
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
    time_per_epoch = time_per_epoch.median() * len(df)

    # Add columns
    df = pd.DataFrame({'Architecture': [f'Lite-HRNet-{arch}'],
                       'Dataset': [dset],
                       'Augmentation': [aug_type],
                       'Pretraining': [pretraining],
                       'Time Per Experiment (Hours)': [round(time_per_epoch.seconds / 60 ** 2, 2)]})

    # Update all_df
    all_df = pd.concat([all_df, df])

# Combine columns
all_df['Dataset, Augmentation, Pretraining'] = \
    all_df['Dataset'].astype(str) + ', ' + all_df['Augmentation'].astype(str) + ', ' + all_df['Pretraining'].astype(str)

# Set seaborn style
sns.set_style('darkgrid')

# Create seaborn plot
g = sns.barplot(data=all_df,
                y='Dataset, Augmentation, Pretraining',
                x='Time Per Experiment (Hours)',
                hue='Architecture',
                order=['Lightbox, None, None',
                       'Lightbox, Geometric, None',
                       'Lightbox, None, MPII',
                       'Lightbox, None, COCO',
                       'Lightbox, Geometric, MPII',
                       'Lightbox, Geometric, COCO',
                       'Sunlamp, None, None',
                       'Sunlamp, Geometric, None',
                       'Sunlamp, None, MPII',
                       'Sunlamp, None, COCO',
                       'Sunlamp, Geometric, MPII',
                       'Sunlamp, Geometric, COCO'])

# Save plot
plt.savefig(root_dir_path.parent / 'Architecture_Investigation_Experiment_Duration_SPEEDplus.png', bbox_inches='tight')

# Close plot
plt.close('all')