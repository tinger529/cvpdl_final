import cv2
import numpy as np
import os
import pickle
from argparse import ArgumentParser
from PIL import Image

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='../cifar-10-batches-py')
    parser.add_argument('--save_dir', type=str, default='./cifar10_dataset')
    args, unknown = parser.parse_known_args()
    return args
 
def unpicke(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')                 # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

    return dic[b'labels'], dic[b'filenames'], dic[b'data']

def main(args):
    for path in os.listdir(args.root):
        if '_batch' not in path: 
            continue
        print(path)

        labels, fnames, datas = unpicke(os.path.join(args.root, path))  
        
        for label, fname, data in zip(labels, fnames, datas):
            label = str(label)
            try:
                os.makedirs(os.path.join(args.save_dir, 'train', label))
                os.makedirs(os.path.join(args.save_dir, 'test', label))
            except:
                pass 
            
            save_data = data.reshape(3, 32, 32).astype('uint8')
            R, G, B = Image.fromarray(save_data[0]), Image.fromarray(save_data[1]), Image.fromarray(save_data[2])
            img = Image.merge('RGB', (R, G, B))
            
            if 'test' in path:
                cv2.imwrite(os.path.join(args.save_dir, 'test', label, str(fname)[2: -1]), np.array(img))
            else:
                cv2.imwrite(os.path.join(args.save_dir, 'train', label, str(fname)[2: -1]), np.array(img))

    return

if __name__ == '__main__':
    main(parse_arguments())