# val_dset_pck_plots.py
# Code to create percentage of correct keypoint (PCK) plots

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 18th September 2023

# Import required modules
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# Path to folder containing all results folders
root_dir_path = Path('/data/mruthven/lite-hrnet/work_dirs')

# SDA type
sda_type = 'Mix'

# List of paths (relative to root_dir_path) to JSON files of results
if sda_type == 'All':
    res_path_list = [Path(f'litehrnet_30_speedplus_640x640_lbx_337_ft_{sda_type.lower()}/20230911_170306.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_337_ft_{sda_type.lower()}_aug/20230911_170514.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_674_ft_{sda_type.lower()}/20230911_225426.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_674_ft_{sda_type.lower()}_aug/20230912_104547.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_1348_ft_{sda_type.lower()}/20230912_134927.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_1348_ft_{sda_type.lower()}_aug/20230912_210737.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_2022_ft_{sda_type.lower()}/20230912_212302.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_2022_ft_{sda_type.lower()}_aug/20230913_044722.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_2696_ft_{sda_type.lower()}/20230913_092824.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_2696_ft_{sda_type.lower()}_aug/20230913_193039.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_4044_ft_{sda_type.lower()}/20230914_173446.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_lbx_4044_ft_{sda_type.lower()}_aug/20230914_212419.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_140_ft_{sda_type.lower()}/20230911_143347.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_140_ft_{sda_type.lower()}_aug/20230911_143801.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_279_ft_{sda_type.lower()}/20230911_205059.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_279_ft_{sda_type.lower()}_aug/20230911_205353.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_558_ft_{sda_type.lower()}/20230912_104504.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_558_ft_{sda_type.lower()}_aug/20230912_155743.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_837_ft_{sda_type.lower()}/20230913_221016.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_837_ft_{sda_type.lower()}_aug/20230914_092725.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_1116_ft_{sda_type.lower()}/20230914_022526.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_1116_ft_{sda_type.lower()}_aug/20230913_140827.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_1675_ft_{sda_type.lower()}/20230914_092503.log.json'),
                    Path(f'litehrnet_30_speedplus_640x640_slp_1675_ft_{sda_type.lower()}_aug/20230914_142142.log.json')]
elif sda_type == 'Mix':
    res_path_list = [Path(f'litehrnet_30_speedplus_640x640_syn_lbx/20230915_103042.log.json'),
                     Path(f'litehrnet_30_speedplus_640x640_syn_lbx_aug/20230915_175411.log.json'),
                     Path(f'litehrnet_30_speedplus_640x640_syn_lbx_aug2/20230915_181755.log.json'),
                     Path(f'litehrnet_30_speedplus_640x640_syn_slp/20230915_120856.log.json'),
                     Path(f'litehrnet_30_speedplus_640x640_syn_slp_aug/20230915_175803.log.json'),
                     Path(f'litehrnet_30_speedplus_640x640_syn_slp_aug2/20230915_181605.log.json')]
    idx_dict = {0: 0, 1: 337, 2: 674, 3: 0, 4: 140, 5: 279}


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

    # Determine number of real images used for fine-tuning
    if sda_type == 'All':
        n_real = str(res_path.parent).split('_')
        n_real = int(n_real[5])
    elif sda_type == 'Mix':
        n_real = idx_dict[idx]
    
    # Read JSON file using pandas
    df = pd.read_json(root_dir_path / res_path, lines=True)

    # Extract results on validation dataset
    df = df[df['mode'] == 'val']

    # Extract relevant columns
    df = df[['epoch', 'PCKh']]

    # Rename columns
    df.rename(columns={'epoch': 'Epoch', 'PCKh': 'Mean Percentage Correct Keypoints'}, inplace=True)

    # Add columns
    df['NrealOverNsyn'] = round(n_real / n_syn, 4)
    df['Dataset'] = dset
    df['Augmentation'] = aug_type

    # Update all_df
    all_df = pd.concat([all_df, df])

# Set seaborn style
sns.set_style('darkgrid')

# Create seaborn plot
g = sns.relplot(data=all_df,
                kind='line',
                x='Epoch',
                y='Mean Percentage Correct Keypoints',
                hue='NrealOverNsyn',
                style='Augmentation',
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
plt.savefig(root_dir_path.parent / f'Lite-HRNet_SDA_Val_Dset_PCK_SPEEDplus_{sda_type}_{datetime.date.today()}.png', bbox_inches='tight')

# Close plot
plt.close('all')