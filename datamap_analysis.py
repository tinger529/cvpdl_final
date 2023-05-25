import glob
import json
import os
import numpy as np
import pandas as pd 
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--datamap_dir', type=str, default='./datamap')
    parser.add_argument('--save_fname', type=str, default='augmentation_comparison.json')
    args, unknown = parser.parse_known_args()
    return args

def main(args):
    datamap_files = glob.glob(f'{args.datamap_dir}/**/*.json')

    aug_delta_dict = dict()

    with open(os.path.join(args.datamap_dir, 'none', 'none.json')) as json_file:
        org_datamap = json.load(json_file)
    org_conf = np.array(org_datamap['gold_prob_means'])
    org_var = np.array(org_datamap['gold_prob_stds'])

    for fname in datamap_files:
        augment_type = fname.split('/')[-1].split('.json')[0]

        if augment_type == 'none': continue

        with open (fname) as json_file:
            datamap = json.load(json_file)

        conf = np.array(datamap['gold_prob_means'])
        var = np.array(datamap['gold_prob_stds'])

        conf_delta_mean = np.mean(conf - org_conf)
        var_delta_mean = np.mean(var - org_var)

        aug_delta_dict[augment_type] = {
            'conf_delta': conf_delta_mean,
            'var_delta': var_delta_mean
        }
    
    with open(os.path.join(args.datamap_dir, args.save_fname), 'w') as writer:
        writer.write(json.dumps(aug_delta_dict, indent=4))
    
    # df = pd.DataFrame(aug_delta_dict)
    # df.to_csv(os.path.join(f'{args.datamap_dir}, {args.save_fname}'))

    return

if __name__ == '__main__':
    main(parse_arguments())