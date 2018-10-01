from __future__ import unicode_literals

import os
import sys
import json
import fnmatch
import tarfile
from PIL import Image
from glob import glob
from tqdm import tqdm
from six.moves import urllib

import numpy as np

from utils import loadmat, imread, imwrite
from scipy.misc import imresize

DATA_FNAME = 'gaze.npz'

def save_array_to_grayscale_image(array, path):
  Image.fromarray(array).convert('L').save(path)

def process_json_list(json_list, img):
  ldmks = [eval(s) for s in json_list]
  return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])

def maybe_preprocess(config, data_path, sample_path=None):
  if config.max_synthetic_num < 0:
    max_synthetic_num = None
  else:
    max_synthetic_num = config.max_synthetic_num

  return os.path.join(data_path, config.synthetic_image_dir),

def load(config, data_path, sample_path, rng):
  if not os.path.exists(data_path):
    print('creating folder', data_path)
    os.makedirs(data_path)

  synthetic_image_path = maybe_preprocess(config, data_path, sample_path)

  return synthetic_image_path

class DataLoader(object):
  def __init__(self, config, rng=None):
    self.rng = np.random.RandomState(1) if rng is None else rng

    self.data_path = os.path.join(config.data_dir, 't18')
    self.batch_size = config.batch_size
    self.debug = config.debug

    real_image_path = os.path.join(self.data_path, 'real')
    synthetic_image_path = os.path.join(self.data_path, config.synthetic_image_dir)

    self.real_data_paths = np.array(glob(os.path.join(real_image_path, '*.jpg')))
    self.synthetic_data_paths = np.array(glob(os.path.join(synthetic_image_path, '*.jpg')))
    self.synthetic_data_dims = list(imread(self.synthetic_data_paths[0], flatten=True).shape) + [1]

    self.synthetic_data_paths.sort()

    self.real_p = 0
    self.config = config

  def reset(self):
    self.real_p = 0

  def __iter__(self):
    return self

  def __next__(self, n=None):
    """ n is the number of examples to fetch """
    if n is None: n = self.batch_size

    if self.real_p == 0:
      inds = self.rng.permutation(self.real_data_paths.shape[0])
      self.real_data_paths = self.real_data_paths[inds]

    if self.real_p + n > self.real_data_paths.shape[0]:
      self.reset()

    x= np.expand_dims(np.stack([imresize(imread(path, flatten=True), (self.config.input_height, self.config.input_width)) for path in self.real_data_paths[self.real_p:self.real_p+n]]), -1)
    self.real_p += self.batch_size

    return x

  next = __next__
