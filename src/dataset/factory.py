
#%%
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import pathlib
import random

data_root = pathlib.Path('./data')

def create_labelled_images():
  # Read in list of image paths
  image_path_list = list(data_root.glob('*/*.png'))
  image_paths = [str(path) for path in image_path_list]
  random.shuffle(image_paths)
  
  # Infer labels from directory names
  labels = sorted(
    item.name
    for item in data_root.glob('*/') if item.is_dir()
  )
  # Create indexed labels for rock, scissors and paper.
  # Each image will receive one of these mapped indices as its
  # category.
  label_with_index = dict(
    (name, index)
    for index, name in enumerate(labels))
  labelled_images = [
    label_with_index[pathlib.Path(path).parent.name]
    for path in image_paths
  ]
  return labelled_images

def create():
  labelled_images = create_labelled_images()
  print('First 10 labelled images: ', labelled_images[:10])
#%%
