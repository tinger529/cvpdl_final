import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import seaborn as sns
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--tfm_type', type=str, default='cutout')
    parser.add_argument('--datamap_dir', type=str, default='./datamap/intel')
    parser.add_argument('--title', type=str, default='Cutout')
    args = parser.parse_args()
    return args

def main(args):
    with open(os.path.join(args.datamap_dir, f'{args.tfm_type}.json')) as json_file:
        data = json.load(json_file)
        data['correct'] = data['correct_means']
    
    sns.set()
    datamap = pd.DataFrame(data)
    pal = sns.diverging_palette(260, 15, n=len(datamap['correct'].unique().tolist()), sep=10, center='dark')

    fig = plt.figure(figsize=(14, 10), )
    sns.scatterplot(x='gold_prob_stds', y='gold_prob_means', data=datamap, palette=pal, hue='correct', s=10)

    plt.title(args.title)
    plt.xlabel('variability')
    plt.ylabel('confidence')
    plt.savefig(os.path.join(args.datamap_dir, f'{args.tfm_type}.png'))

if __name__ == '__main__':
    main(parse_arguments())