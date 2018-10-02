from __future__ import unicode_literals

import sys
import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config

config = None

def main(_):
  tf.set_random_seed(config.random_seed)
  prepare_dirs(config)

  trainer = Trainer(config)
  save_config(config.model_dir, config)

  if config.is_train:
    trainer.train()
  else:
    trainer.test()

if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
