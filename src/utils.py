import argparse
import os
import json
from datetime import datetime
import logging
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import Dataset
import numpy as np

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()


def Initialization(args):
    """
            Input:
                args: arguments object from argparse
            Returns:
                config: configuration dictionary
    """

    config = args.args.__dict__  # configuration dictionary
    config['dataset'] = problem
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_path']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, config['dataset'],
                              initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    config['device'] = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(config['device']))
    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def save_metrics_to_excel(filepath, all_metrics):
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        # Convert targets to a dataframe and write to the first sheet
        df_targets = pd.DataFrame(all_metrics['targets'], columns=['Target'])
        df_targets.to_excel(writer, sheet_name='Targets', index=False)
        
        # Convert predictions to a dataframe and write to the second sheet
        df_predictions = pd.DataFrame(all_metrics['predictions'], columns=['Predictions'])
        df_predictions.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Convert probabilities (probs) to a dataframe and write to the third sheet
        df_probs = pd.DataFrame(all_metrics['probs'], columns=[f'Prob_{i}' for i in range(all_metrics['probs'].shape[1])])
        df_probs.to_excel(writer, sheet_name='Probabilities', index=False)


def save_model(path, epoch, model, optimizer=None):
    # Ensure the directory exists before saving the model
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, path):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            save_model(path, epoch, model, optimizer)


class SaveBestACCModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_acc=float('0')):
        self.best_valid_acc = best_valid_acc

    def __call__(self, current_valid_acc, epoch, model, optimizer, criterion, path):

        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            print(f"Best validation acc: {self.best_valid_acc}")
            print(f"Saving best model for epoch: {epoch}\n")
            save_model(path, epoch, model, optimizer)


def load_model(model, model_path, optimizer=None, resume=False, change_output=False,
               lr=None, lr_step=None, lr_factor=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    if change_output:
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('output_layer'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)
    return config


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


class dataset_class(Dataset):

    def __init__(self, data, label):
        super(dataset_class, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)
        # self.__padding__()

    def __padding__(self):
        origin_len = self.feature[0].shape[1]
        if origin_len % self.patch_size:
            padding_len = self.patch_size - (origin_len % self.patch_size)
            padding = np.zeros((len(self.feature), self.feature[0].shape[0], padding_len), dtype=np.float32)
            self.feature = np.concatenate([self.feature, padding], axis=-1)

    def __getitem__(self, ind):

        x = self.feature[ind]
        x = x.astype(np.float32)
        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)