import os
import numpy as np
import logging
from sklearn import model_selection
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)



def load(config, fold):
    dataset = config['dataset']
    repo_id = f"monster-monash/{dataset}"

    # Download data
    data_path = hf_hub_download(repo_id=repo_id, filename=f"{dataset}_X.npy", repo_type="dataset")
    Data_npy = np.load(data_path, mmap_mode="r")  # (#Samples, #Channel, #Length)

    # Download labels
    label_filename = f"{dataset}_Y.npy"
    try:
        label_path = hf_hub_download(repo_id=repo_id, filename=label_filename, repo_type="dataset")
    except:
        label_filename = f"{dataset}_y.npy"
        label_path = hf_hub_download(repo_id=repo_id, filename=label_filename, repo_type="dataset")
    Label_npy = np.load(label_path)
    # Load test indices
    try:
        test_index_path = hf_hub_download(repo_id=repo_id, filename=f"test_indices_fold_{fold}.txt", repo_type="dataset")
        test_index = np.loadtxt(test_index_path, dtype=int)
    except Exception as e:
        logger.error(f"Failed to load test indices: {e}")
        raise
    Data = split_data(Data_npy, Label_npy, test_index)

    logger.info("{} samples will be used for training ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))
    return Data


def split_data(Data_npy, Label_npy, test_index):

    # Create a boolean array indicating the samples designated for the test set
    test_bool_index = np.zeros(len(Label_npy), dtype=bool)
    test_bool_index[test_index] = True

    Data = {'test_data': Data_npy[test_index], 'test_label': Label_npy[test_index],
            'All_train_data': Data_npy[~test_bool_index], 'All_train_label': Label_npy[~test_bool_index]}

    '''
        if 'subject_id' in Meta_data.item():
        Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = (
            subject_wise_split(Data_npy, Label_npy, Meta_data, test_index))
        Data['train_data'] = np.concatenate((Data['train_data'], Data['val_data']))
        Data['train_label'] = np.concatenate((Data['train_label'], Data['val_label']))
        
    else:
    '''

    Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = (
        non_subject_wise_split(Data['All_train_data'], Data['All_train_label']))
    return Data


def non_subject_wise_split(data, label):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]

    return train_data, train_label, val_data, val_label



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
