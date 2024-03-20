# create_complete_eval_metric_plot.py
# Code to create plots of evaluation metrics

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 7th September 2023

# Import required modules
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


# Dictionary mapping each experiment to the correct JSON file
exp_dict = {'Sunlamp1': (Path('work_dirs/litehrnet_30_speedplus_640x640_slp_new'), 'None'),
            'Sunlamp2': (Path('work_dirs/litehrnet_30_speedplus_640x640_slp_new_aug'), 'Geometric')}

# Dictionary mapping file name abbreviations
abr_dict = {'train': 'Training',
            'val': 'Validation',
            'test': 'Test'}

# Preallocate pandas DataFrame for resuts
all_df = pd.DataFrame()

# For each file
for exp_name, (file_path, aug_type) in exp_dict.items():

    # For training, validation and test datasets
    for dset_split, full_name in abr_dict.items():
    
        # Read CSV file using pandas
        df = pd.read_csv(file_path / f'{dset_split}_dataset_per_image_accuracy.csv')
    
        # Add experiment info
        df['Dataset'] = exp_name[:-1]

        # Add augmentation info
        df['Augmentation'] = aug_type

        # Add dtaset split
        df['Split'] = full_name

        # Update all_df
        all_df = pd.concat([all_df, df])
        
# Rename columns
all_df.rename(columns={'PCK': 'Percentage Correct Keypoints',
                       'MeanKLE': 'Mean Keypoint Location Error'}, inplace=True)

# Set seaborn style
sns.set_style('darkgrid')

# Create seaborn plot
g = sns.histplot(data=all_df[all_df['Augmentation']=='None'],
                 x='Percentage Correct Keypoints',
                 hue='Split',
                 stat='probability',
                 bins=10).set(title='Sunlamp (No Augmentation)')

# Save plot
plt.savefig('slp_no_aug_pck_hist.png', bbox_inches='tight')

# Close plot
plt.close('all')

# Create seaborn plot
g = sns.histplot(data=all_df[all_df['Augmentation']=='Geometric'],
                 x='Percentage Correct Keypoints',
                 hue='Split',
                 stat='probability',
                 bins=10).set(title='Sunlamp (Geometric Augmentations)')

# Save plot
plt.savefig('slp_geo_aug_pck_hist.png', bbox_inches='tight')

# Close plot
plt.close('all')

# Create seaborn plot
g = sns.histplot(data=all_df[all_df['Split']=='Training'],
                 x='Percentage Correct Keypoints',
                 hue='Augmentation',
                 bins=10).set(title='Sunlamp (Training Dataset)')

# Save plot
plt.savefig('slp_train_pck_hist.png', bbox_inches='tight')

# Close plot
plt.close('all')

# Rearrange all_df
tmp_df = all_df[all_df['Split']=='Training'].pivot(columns='Augmentation',values='Percentage Correct Keypoints')
tmp_df['PCK Difference'] = tmp_df['Geometric'] - tmp_df['None']

# Create seaborn plot
g = sns.histplot(data=tmp_df,
                 x='PCK Difference',
                 bins=10).set(title='Sunlamp (Training Dataset)')
