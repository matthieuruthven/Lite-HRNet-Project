# architecture_investigation.py
# Code to create percentage of correct keypoint (PCK) plots
# to enable investigation into effect of HRNet architecture
# on Lite-HRNet accuracy

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 25th September 2023

# Import required modules
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

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

# Number of synthetic images
n_syn = 47966

# For each file
for idx, res_path in enumerate(res_path_list):
                
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
    
    # Read JSON file using pandas
    df = pd.read_json(root_dir_path / res_path, lines=True)

    # Extract results on validation dataset
    df = df[df['mode'] == 'val']

    # Extract relevant columns
    df = df[['epoch', 'PCKh']]

    # Add columns
    df['Architecture'] = f'Lite-HRNet-{arch}'
    df['Dataset'] = dset
    df['Augmentation'] = aug_type
    df['Pretraining'] = pretraining

    # Update all_df
    all_df = pd.concat([all_df, df])

# Rename columns
all_df.rename(columns={'epoch': 'Epoch', 'PCKh': 'Mean Percentage Correct Keypoints'}, inplace=True)

# Create new column
all_df['Architecture (Augmentation)'] = all_df['Architecture'].astype(str) + ' (' + all_df['Augmentation'].astype(str) + ')'

# Set seaborn style
sns.set_style('darkgrid')

# Create seaborn plot
g = sns.relplot(data=all_df,
                kind='line',
                x='Epoch',
                y='Mean Percentage Correct Keypoints',
                hue='Architecture (Augmentation)',
                style='Pretraining',
                col='Dataset',
                col_order=['Lightbox', 'Sunlamp'],
                errorbar=None)

# Modify axis titles
counter = 0
for ax in g.axes.flatten():
    if (counter == 0):
        ax.set_title("Lightbox")
    elif (counter == 1):
        ax.set_title("Sunlamp")
    counter += 1

# Save plot
plt.savefig(root_dir_path.parent / f'Architecture_Investigation_Val_Dset_PCK_SPEEDplus_{datetime.date.today()}.png', bbox_inches='tight')

# Close plot
plt.close('all')

# Create seaborn plot
g = sns.relplot(data=all_df,
                kind='line',
                x='Epoch',
                y='Mean Percentage Correct Keypoints',
                hue='Architecture',
                style='Pretraining',
                col='Augmentation',
                col_order=['None', 'Geometric'],
                row='Dataset',
                row_order=['Lightbox', 'Sunlamp'],
                errorbar=None)

# Save plot
plt.savefig(root_dir_path.parent / f'Architecture_Investigation_Val_Dset_PCK_SPEEDplus_{datetime.date.today()}_pretraining_effect.png', bbox_inches='tight')

# Close plot
plt.close('all')

# Create seaborn plot
g = sns.relplot(data=all_df,
                kind='line',
                x='Epoch',
                y='Mean Percentage Correct Keypoints',
                hue='Architecture',
                style='Augmentation',
                col='Pretraining',
                col_order=['None', 'MPII', 'COCO'],
                row='Dataset',
                row_order=['Lightbox', 'Sunlamp'],
                errorbar=None)

# Save plot
plt.savefig(root_dir_path.parent / f'Architecture_Investigation_Val_Dset_PCK_SPEEDplus_{datetime.date.today()}_aug_effect.png', bbox_inches='tight')

# Close plot
plt.close('all')