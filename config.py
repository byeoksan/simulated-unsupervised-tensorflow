#-*- coding: utf-8 -*-
import argparse
import utils
import os

def str2bool(v):
  return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_set', type=str, default='t18')
data_arg.add_argument('--data_dir', type=str, default='data')
data_arg.add_argument('--real_image_dir', type=str, default="real")
data_arg.add_argument('--synthetic_image_dir', type=str, default="syn")
data_arg.add_argument('--input_height', type=int, default=88)
data_arg.add_argument('--input_width', type=int, default=148)
data_arg.add_argument('--max_synthetic_num', type=int, default=-1)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--max_step', type=int, default=20000, help='')
train_arg.add_argument('--reg_scale', type=float, default=0.5, help='')
train_arg.add_argument('--initial_K_d', type=int, default=200, help='')
train_arg.add_argument('--initial_K_g', type=int, default=1000, help='')
train_arg.add_argument('--K_d', type=int, default=1, help='')
train_arg.add_argument('--K_g', type=int, default=2, help='')
train_arg.add_argument('--batch_size', type=int, default=64, help='')
train_arg.add_argument('--buffer_size', type=int, default=20000, help='')
train_arg.add_argument('--num_epochs', type=int, default=12, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--learning_rate', type=float, default=0.0001, help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')
train_arg.add_argument('--max_grad_norm', type=float, default=50, help='')
train_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--model_name', type=str, required=True, help='Name of the model')
misc_arg.add_argument('--log_step', type=int, default=20, help='')
misc_arg.add_argument('--debug', type=str2bool, default=False)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
misc_arg.add_argument('--max_image_summary', type=int, default=7)
misc_arg.add_argument('--sample_image_grid', type=eval, default='[8, 8]')

def get_config():
  config, unparsed = parser.parse_known_args()

  config.model_dir = config.model_name
  config.log_dir = os.path.join(config.model_dir, 'logs')
  config.sample_dir = os.path.join(config.model_dir, 'samples')
  config.output_dir = os.path.join(config.model_dir, 'outputs')

  return config, unparsed
