"""
Author: Mason Hu
------------------------------------------------------------------

This is the main script for model training, which make function calls
to the sub modules for each stage of training.

------------------------------------------------------------------
"""

# Standard lib imports
import json
import os

# 3rd party imports
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import numpy as np

# Project level imports
from src.preprocessing import preprocessing
from src.data_loader import data_loading, get_default_device, to_device
from src.model import CharCNN
from src.pipeline import pipeline_process
from src.general_util import stratified_shuffle


#########################################################################
# Load config file for dir reference

fp = os.path.join
with open("config.json", 'r') as conf_file:
    config = json.load(conf_file)
    config['root_dir'] = os.getcwd()
    for k, v in config.items():
        if k != 'root_dir':
            config[k] = fp(config['root_dir'], fp(*v))

#########################################################################
# Training parameter definition


def get_default_params():

    data_paths = {
        "raw_path": config['raw_path'],
        "train_path": config['train_path'],
        "test_path": config['test_path']
    }

    # Data preprocessing parameters
    prep_params = {
        "infer": False,  # Set this to False for training, and True if making inference
        "chunksize": 100_000,  # Batch processing chunk size when reading in raw scraped data
        "encoding": 'utf-8',  # Type of encoding for reading in data from csv files
        "text_col": 'comment',  # The name of text column in the raw csv data
        "label_col": 'rating',  # The name of label column in the raw csv data
        "train_size": 0.9,  # The size of training set for train test split
        "random_state": 10,  # The random state seed for train test split, set for reproducible results
        "class_data_usage": [1, 1, 0.16],  # Specify the proportion of raw data within each class to be used
        "classes": [0, 1, 2]  # The distinct classes within the raw data label
    }

    # Text quantisation parameters
    quant_params = {
        # the alphabet to be used for text quantisation.
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}\n",
        "max_length": 1014  # max_length: the maximum text input length.
    }

    # Data Loading parameters
    data_params = {
        "train_params": {
            "batch_size": 1024,    # The batch size of preprocessed data to be loaded during training
            # Random shuffling of data during batch loading, note this should be false if sampler is True
            "shuffle": False,
            # Implement Weighted random sampler to reduce class imbalance during batch loading
            "sampler": True,
            "num_workers": 2,     # Number of threads used during training
            # Pinning memory especially for GPU training (good when input data is fixed size)
            "pin_memory": True,
            # If last batch loaded does not match batch_size specified then drop it
            "drop_last": True
        },
        "test_params": {
            "batch_size": 1024,
            "shuffle": False,
            "sampler": False,
            "num_workers": 2,
            "pin_memory": True,
            "drop_last": True
        }
    }

    # Model parameters (Feels a bit ugly...)
    """
    *Note: It is important to load a model with the same
    dimensional parameters it was trained on.
    """
    model_params = {
        # Used to calculate the number of input features to the first linear layer in the model
        "batch_size": data_params['train_params']['batch_size'],
        # Used to calculate number of input channels to the first conv1d layer in the model
        "num_chars": len(quant_params['alphabet']),
        # Same purpose as batch_size
        "max_length": quant_params['max_length'],
        # Determines the number of output units in the last layer of the model
        "num_classes": 3
    }

    # Pipeline parameters
    pipe_params = {
        "optimiser": 'sgd',                       # Choose optimiser type, sgd or adam
        "unbalance_classes": False,               # Set to true if label classes are imbalanced
        "lr": 0.01,                               # Learning rate for gradient descent
        "momentum": 0.9,                          # momentum (sgd) coeff
        "schedule_lr": True,                      # Learning rate scheduling
        "num_epoch": 10,                          # Number of epochs to train
        "log_file": config['log_file'],           # Path to log file, loggin for every 'print_every' iterations
        "log_f1": True,                           # Print out F1 score for every 'print_every' batch iterations
        "print_every": 50,                        # Print out training metrics after this amount of iterations within an epoch
        "checkpoint": True,                       # Save intermediate models based on f1 scoring.
        "model_folder": config['model_path'],     # Folder to save intermediate models
        "model_prefix": "20200624",               # Folder
        "early_stop": True,                       # Early stopping if evaluation metrics don't improve after 'patience' epochs
        "patience": 3                             # Number of epochs before early stop is enforced
    }

    return (data_paths, prep_params, quant_params,
            data_params, model_params, pipe_params)

    """
    TODO:
    - Implement parameter inputs using commandline options.
    - Implement parameter inputs using text file
    """

###############################################################################
# Main


if __name__ == "__main__":

    all_params = get_default_params()
    data_paths = all_params[0]
    prep_params = all_params[1]
    quant_params = all_params[2]
    data_params = all_params[3]
    model_params = all_params[4]
    pipe_params = all_params[5]

    # Setup tensorboard
    writer = SummaryWriter(config['log_path'])

    print("Preparing Data")
    train_df, test_df = preprocessing(data=None, **data_paths, **prep_params)

    # Stratify shuffle the test set to ensure each batch during validation will
    # contain all of the classes as per proportion in the original test set
    test_df = stratified_shuffle(test_df, model_params['batch_size'],
                                prep_params['label_col'], 10)

    # Data loading
    train_dl, test_dl = data_loading(train_df, test_df,
                                    prep_params['text_col'],
                                    prep_params['label_col'],
                                     **quant_params,
                                     **data_params)

    print("Start Training")
    # Model instantiation
    device = get_default_device()
    print(f"Using device: {device}")
    model = CharCNN(*tuple([val for val in model_params.values()]))
    # Add visualisation of model to tensorboard
    input_graph = to_device(Tensor(np.random.randn(model_params['batch_size'],
                                                    model_params['max_length'],
                                                    model_params['num_chars'])), device)
    writer.add_graph(model, input_graph)

    # Training
    pipeline_process(model, train_dl, test_dl,
                    train_df['rating'],
                    quant_params['max_length'],
                    writer,
                    **pipe_params)

    writer.close()
    