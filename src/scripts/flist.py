# script flist.py is modified from below github reference link
# Reference link: https://github.com/knazeri/edge-connect/blob/master/scripts/flist.py

import os
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--output', type=str, help='path to the file list')
parser.add_argument('--test_size', type=int, help="only use this argument when generating flist of testing images")
args = parser.parse_args()

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

images = []
for root, dirs, files in os.walk(args.path):
    if args.test_size:
        if "Places" in root or "Celeba" in root:
            files = random.sample(files, args.test_size//2)

    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')