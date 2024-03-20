# create_complete_eval_metric_plot_slp.py
# Code to create plots of evaluation metrics

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 11th September 2023

# Import required modules
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


# Dictionary mapping each experiment to the correct JSON file
exp_dict = {'None': Path('work_dirs/litehrnet_30_speedplus_640x640_slp_new/20230906_121820.log.json'),
            'Geometric': Path('work_dirs/litehrnet_30_speedplus_640x640_slp_new_aug/20230906_195216.log.json'),
            'Gaussian Noise': Path('work_dirs/litehrnet_30_speedplus_640x640_slp_new_aug_noise/20230908_150553.log.json'),
            'Coarse Dropout': Path('work_dirs/litehrnet_30_speedplus_640x640_slp_new_aug_dropout/20230908_222843.log.json')}

# Preallocate pandas DataFrame for resuts
all_df = pd.DataFrame()

# For each file
for exp_name, json_path in exp_dict.items():

    # Read JSON file using pandas
    df = pd.read_json(json_path, lines=True)

    # Extract results on training dataset
    tmp_df = df[df['mode'] == 'train']

    # Extract relevant columns
    tmp_df = tmp_df[['epoch', 'iter', 'heatmap_loss', 'acc_pose']]

    # Maximum iteration
    max_iter = tmp_df['iter'].max()

    # List of unique values in iter column
    iter_list = tmp_df['iter'].unique()

    # Iterations interval
    iter_interval = (iter_list[-1] - iter_list[0]) / (len(iter_list) - 1)

    # New Epoch column
    tmp_df['Epoch'] = (tmp_df['epoch'] + tmp_df['iter'] / (max_iter + iter_interval))

    # Convert proportion to percentage
    tmp_df['acc_pose'] *= 100

    # Rename columns
    tmp_df.rename(columns={'heatmap_loss': 'Mean Squared Error',
                            'acc_pose': 'Mean Percentage Correct Keypoints'}, inplace=True)
    
    # Add augmentation info
    tmp_df['Augmentation'] = exp_name

    # Add dtaset split
    tmp_df['Split'] = 'Training'

    # Update all_df
    all_df = pd.concat([all_df, tmp_df])

    # Extract results on validation dataset
    tmp_df = df[df['mode'] == 'val']

    # Extract relevant columns
    tmp_df = tmp_df[['epoch', 'PCKh']]

    # Rename columns
    tmp_df.rename(columns={'epoch': 'Epoch', 'PCKh': 'Mean Percentage Correct Keypoints'},
                    inplace=True)
    
    # Add experiment info
    tmp_df['Augmentation'] = exp_name

    # Add dtaset split
    tmp_df['Split'] = 'Validation'

    # Update all_df
    all_df = pd.concat([all_df, tmp_df])
        
# Set seaborn style
sns.set_style('darkgrid')

# Create seaborn plot
g = sns.lineplot(data=all_df,
                 x='Epoch',
                 y='Mean Percentage Correct Keypoints',
                 hue='Augmentation',
                 hue_order=['None', 'Geometric', 'Gaussian Noise', 'Coarse Dropout'],
                 style='Split',
                 style_order=['Training', 'Validation'])

# Show plot
# plt.show()

# Save plot
plt.savefig(f'slp_train_val_pck_plot_{datetime.date.today()}.png', bbox_inches='tight')