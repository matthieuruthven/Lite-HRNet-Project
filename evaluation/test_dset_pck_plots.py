# test_dset_pck_plots.py
# Code to create percentage of correct keypoint (PCK) plots

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 15th September 2023

# Import required modules
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# Path to folder containing all results folders
root_dir_path = Path('/data/mruthven/lite-hrnet/work_dirs')

# List of folders containing results files
res_dir_list = ['litehrnet_30_speedplus_640x640_lbx_337_ft_all_aug',
                'litehrnet_30_speedplus_640x640_lbx_674_ft_all_aug',
                'litehrnet_30_speedplus_640x640_lbx_1348_ft_all_aug',
                'litehrnet_30_speedplus_640x640_lbx_2022_ft_all_aug',
                'litehrnet_30_speedplus_640x640_lbx_2696_ft_all_aug',
                'litehrnet_30_speedplus_640x640_lbx_4044_ft_all_aug',
                'litehrnet_30_speedplus_640x640_slp_140_ft_all_aug',
                'litehrnet_30_speedplus_640x640_slp_279_ft_all_aug',
                'litehrnet_30_speedplus_640x640_slp_558_ft_all_aug',
                'litehrnet_30_speedplus_640x640_slp_837_ft_all_aug',
                'litehrnet_30_speedplus_640x640_slp_1116_ft_all_aug',
                'litehrnet_30_speedplus_640x640_slp_1675_ft_all_aug']

# Preallocate pandas DataFrame for results
all_df = pd.DataFrame()

# Number of synthetic images
n_syn = 47966

# For each file
for res_dir in res_dir_list:
                
    # Determine dataset
    if 'lbx' in res_dir:
        dset = 'Lightbox'
    elif 'slp' in res_dir:
        dset = 'Sunlamp'

    # Determine number of real images used for fine-tuning
    n_real = res_dir.split('_')
    try:
        n_real = int(n_real[5])
    except:
        n_real = 0

    # Detemine SDA approach
    if 'ft_all' in res_dir:
        sda_type = '2-Stage (Entire Model Fine-Tuned)'
    else:
        sda_type = '1-Stage (Syn and Real Data Mixed)'

    # Read CSV file of results
    df = pd.read_csv(root_dir_path / res_dir / 'test_dataset_per_image_accuracy.csv')

    # Add columns
    df['NrealOverNsyn'] = round(n_real / n_syn, 4)
    df['Dataset'] = dset
    df['SDA Approach'] = sda_type

    # Add image IDs
    df = df.reset_index()

    # Update all_df
    all_df = pd.concat([all_df, df])

# Remove unnecessary columns
all_df = all_df[['Dataset', 'SDA Approach', 'index', 'PCK', 'MeanKLE', 'NrealOverNsyn']]

# Melt PCK and KLE columns
all_df = pd.melt(all_df, id_vars=['Dataset', 'SDA Approach', 'index', 'NrealOverNsyn'], value_vars=['PCK', 'MeanKLE'])

# Rename columns
all_df.rename(columns={'variable': 'Evaluation Metric',
                       'value': 'Value'}, inplace=True)

# Set seaborn style
sns.set_style('darkgrid')

# Create seaborn plot
g = sns.relplot(data=all_df,
                kind='line',
                marker='o',
                x='NrealOverNsyn',
                y='Value',
                hue='Dataset',
                style='SDA Approach',
                col='Evaluation Metric',
                errorbar=None,
                facet_kws={'sharey': False, 'sharex': True})

# Remove plot titles
g.set_titles("")

# Modify axis labels
counter = 0
for ax in g.axes.flatten():
    ax.set_xlabel("$N_{real}$ / $N_{syn}$")
    if counter == 0:
        ax.set_ylabel("Mean Proportion of Correct Keypoints")
    elif counter == 1:
        ax.set_ylabel("Mean Keypoint Location Error in Pixels")
    counter += 1

# Increase spacing between subplots
g.figure.subplots_adjust(wspace=0.2, hspace=0)

# Save plot
plt.savefig(root_dir_path.parent / f'Lite-HRNet_SDA_Test_Dset_PCK_KLE_SPEEDplus_{datetime.date.today()}.png', bbox_inches='tight')

# Close plot
plt.close('all')