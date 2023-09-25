#
import os
import glob
import shutil
import numpy as np
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
import cv2
import json
from PIL import Image


ROOT = "./"


# get names of all files in directory
list_of_files = glob.glob(os.path.join(ROOT, 'data/things/*/*.jpg'))
r_indx = np.random.randint(0, len(list_of_files), 64)
list_of_files = [list_of_files[i] for i in r_indx]
# list_of_prompts = [list_of_prompts[i] for i in r_indx]

with open(os.path.join(ROOT, 'data/things/val_data.json'), 'wt') as f:
    for i in range(len(list_of_files)):
        object_name = list_of_files[i].split('/')[-2]
        f.write(json.dumps({'target': list_of_files[i].replace(object_name, 'images'), 'source': list_of_files[i].replace(object_name, 'edges'), 'prompt': object_name, 'ds_label': 'things'}) + '\n')


os.makedirs(os.path.join(ROOT, 'data/things/edges/'), exist_ok=True)
apply_canny = CannyDetector()

for i in range(len(list_of_files)):
    object_name = list_of_files[i].split('/')[-2]
    target = cv2.imread(list_of_files[i])
    target = resize_image(target, 256)
    detected_map = apply_canny(target, 100, 200)
    Image.fromarray(detected_map).save(list_of_files[i].replace(object_name, 'edges'))
    # detected_map = HWC3(detected_map)

    Image.fromarray(target).save(list_of_files[i].replace(object_name, 'images'))






