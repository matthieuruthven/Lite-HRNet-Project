# create_evaluation_metric_plot.py
# Code to create plot of evaluation metric

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 30th August 2023

# Import required modules
import argparse
from pathlib import Path
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


def main(filepath, split):
    """Arguments:
            filepath (Path): path to JSON file of evaluation metric values
            split (string): dataset (either validation or test) from which evaluation metric values were obtained
        Output:
            PNG file of 
        """
    
    # Read JSON file using pandas
    df = pd.read_json(filepath, lines=True)
    
    # If training dataset was evaluated
    if split == 'training':

        # Extract relevant rows of JSON files
        df = df[df['mode'] == 'train']

        # Extract relevant columns
        df = df[['epoch', 'iter', 'heatmap_loss', 'acc_pose']]

        # Maximum iteration
        max_iter = df['iter'].max()

        # List of unique values in iter column
        iter_list = df['iter'].unique()

        # Iterations interval
        iter_interval = (iter_list[-1] - iter_list[0]) / (len(iter_list) - 1)

        # New Epoch column
        df['Epoch'] = (df['epoch'] + df['iter'] / (max_iter + iter_interval))

        # Convert proportion to percentage
        df['acc_pose'] *= 100

        # Rename columns
        df.rename(columns={'heatmap_loss': 'Mean Squared Error',
                           'acc_pose': 'Mean Percentage Correct Keypoints'}, inplace=True)
        
        # Set seaborn style
        sns.set_style('darkgrid')

        # Create seaborn plot
        g = sns.lineplot(data=df,
                         x='Epoch',
                         y='Mean Percentage Correct Keypoints')
        
        # Show plot
        # plt.show()

        # Save plot
        plt.savefig(filepath.parent / f'train_pck_plot_{datetime.date.today()}.png', bbox_inches='tight')

        # Close plot
        plt.close('all')

        # Create seaborn plot
        g = sns.lineplot(data=df,
                         x='Epoch',
                         y='Mean Squared Error')
        
        # Show plot
        # plt.show()

        # Save plot
        plt.savefig(filepath.parent / f'train_mse_plot_{datetime.date.today()}.png', bbox_inches='tight')

    # If validation dataset was evaluated
    if split =='validation':

        # Extract relevant rows of JSON files
        df = df[df['mode'] == 'val']

        # Extract relevant columns
        df = df[['epoch', 'PCKh']]

        # Rename columns
        df.rename(columns={'epoch': 'Epoch', 'PCKh': 'Mean Percentage Correct Keypoints'},
                  inplace=True)
        
        # Convert values in Epoch column to integers
        df['Epoch'] = df['Epoch'].astype(int)
        
        # Set seaborn style
        sns.set_style('darkgrid')

        # Create seaborn plot
        g = sns.lineplot(data=df,
                         x='Epoch',
                         y='Mean Percentage Correct Keypoints')
        
        # Show plot
        # plt.show()

        # Save plot
        plt.savefig(filepath.parent / f'val_pck_plot_{datetime.date.today()}.png', bbox_inches='tight')

        
if __name__ == '__main__':

    # Create argument parser
    parser = argparse.ArgumentParser(description='Code to create a plot of evaluation metrics')

    # Add argument
    parser.add_argument('--filepath',
                        type=Path,
                        required=True,
                        help='Path to JSON file of evaluation metric values')
    parser.add_argument('--split',
                        type=str,
                        required=True,
                        choices=['training', 'validation', 'test'],
                        help='Specify if validation or test dataset was evaluated')
    
    # Parse arguments
    args = parser.parse_args()

    # Check arguments
    assert os.path.exists(args.filepath), f'{args.filepath} does not exist'
    assert args.filepath.suffix == '.json', f'File of evaluation metric values must be JSON'

    # Main function
    main(args.filepath, args.split)
