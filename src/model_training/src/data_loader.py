"""
Author: Mason Hu
------------------------------------------------------------------

This script contains all relevant functions and classes for data loading 
functionalities.

------------------------------------------------------------------
"""

# Standard lib imports
from typing import Union, List, Tuple

# 3rd party imports
from pandas import DataFrame, Series, concat
from numpy import identity, zeros
from torch import Tensor, cuda, device
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Project level imports
from src.general_util import get_sampler
import pdb


####################################################################################################
# Text data conversion

class TextDataset(Dataset):
    """
    Custom class created for converting a raw text/label dataset into quantised
    version of itself, and storing relevant textual attributes. This class subclass 
    torch.utils.data.Dataset with the __len__ method overriden to return the number 
    of text inputs, and the __getitem__ method overriden to return the quantised 
    version of the text data elements and their labels. The dimension of the returned 
    quantised text is directly related to the class attributes i.e number of characters
    in the alphbet and maximum length of text input. These class attributes have default 
    values and may be altered by the user.
    
    *Note: the alphabet and max_length class attributes can be considered as hyperparameters 
    for the model training process. If these parameters are to be altered for model 
    performance experiments, the user may change them at a class level.
    
    :param alphabet: the alphabet to be used for text quantisation. It is critical to escape 
    special characters.
    :type alphabet: str
    :param max_length: the maximum text input length used for either truncating or padding 
    of the quantised text matrix.
    :type max_length: int
    """
    # Class attributes
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}\n"
    max_length = 1014
    identity_mat = DataFrame(identity(len(alphabet)), dtype='float32', 
                             index=list(alphabet), columns=list(alphabet))
    
    
    def __init__(self, texts: Series, labels: Series):
        """
        Class constructor taking texts and corresponding labels as inputs
        :param texts: the raw text data
        :type texts: Series
        :param labels: the corresponding labels to the raw text data.
        :type lables: Series
        """
        self.raw_texts = texts
        self.raw_labels = labels
        self.length = texts.shape[0]

        
    def __len__(self):
        return self.length
    
    
    def __getitem__(self, index):
        """
        Overiding the __getitem__ method. Return quantised version of an text
        element together with its label. The intention of this function is to turn 
        the custom dataset into an "map-style dataset" which can be passed to a 
        torch.utils.data.DataLoader object to allow batch processing further down 
        the pipeline.
        """
        raw_text = self.raw_texts.iloc[index]
        label = self.raw_labels.iloc[index]
        quantised_text = TextDataset.quant_text(raw_text).to_numpy()
        return Tensor(quantised_text), label
    
    
    @classmethod
    def quant_text(cls, data: Union[str, Series]) -> Union[DataFrame, Series]:
        """
        Takes an input text set (can be a single text input or a series of text inputs), 
        and returns the quantised version of all its elements. This method may be called 
        at a class level, which will allow external data outside of the instance to be 
        pass as input to this method. This will be particularly usedful during inference 
        when text must be quantised first.

        :param data: text input on which quantisation shall be performed.
        :type data: Union[str, Series]
        :return: quantised text.
        :rtype: Union[DataFrame, Series[DataFrame]]
        """

        def quant_single_text(data: str):
            text_mat = DataFrame(zeros((cls.max_length, len(cls.alphabet))), 
                                 columns=list(cls.alphabet))
            # Catch scenarios where text is an empty string, do nothing in this case
            if len(data) > 0:
                # pdb.set_trace()
                # Character vectors need to be constructed in reverse order
                # This is to feed data into the CNN in the right order.
                reversed_text = data[::-1]
                # Create quantised text matrix with:
                # columns: characters in alphabet
                # rows: characters in input string, fixed size to max_length
                try:
                    # text_mat dimension: 70 x num_input_chars
                    text_mat = concat([cls.identity_mat.loc[c] for c in reversed_text 
                                       if c in cls.alphabet], axis=1)
                except Exception as e:
                    # text_mat dimension: 70 x 1014
                    text_mat = DataFrame(zeros((len(cls.alphabet), cls.max_length)), 
                                         index=list(cls.alphabet))
                    print(f"Batch failed to load due to {e}. Quantisation could not complete, moving on to the next batch")
                # text_mat dimension: 1014 x 70 (if quantisation failed) else num_input_chars x 70
                text_mat = text_mat.transpose().reset_index(drop=True)
                # Truncate text matrix if input text length is more than max length
                if text_mat.shape[0] > cls.max_length:
                    text_mat = text_mat.iloc[:cls.max_length] # text_mat dimension: 1014 x 70

                # Pad text matrix if input text length is less than max length
                elif 0 < text_mat.shape[0] < cls.max_length:
                    diff = cls.max_length - text_mat.shape[0]
                    df_to_append = DataFrame(zeros((diff, len(cls.alphabet))), columns=list(cls.alphabet))
                    text_mat = concat([text_mat, df_to_append], axis=0).reset_index(drop=True) # text_mat dimension: 1014 x 70
                        
            return text_mat # text_mat dimension: 1014 x 70

        # If input is a single text, then output its corresponding quantised matrix
        if isinstance(data, str):
            return quant_single_text(data)

        # If input is a series of texts, then output a series with corresponding quantised matrices
        elif isinstance(data, Series):
            return data.apply(quant_single_text)

        else:
            raise TypeError("""Allowed types are one of the following: 
                - str
                - pandas.Series[str]""")
            
            
####################################################################################################
# Data manipulation for training with GPU

def get_default_device() -> device:
    """
    Pick GPU as device for data if available, else CPU.
    """
    if cuda.is_available():
        return device('cuda')
    else:
        return device('cpu')

    
def to_device(data: Union[Tensor, List, Tuple], 
              device: device) -> Tensor:
    """
    Recursively move tensor(s) to chosen device.
    
    :param data: Input tensor(s)
    :type data: Union[Tensor, List, Tuple]
    :param device: Device to which the input data must be moved. GPU or CPU.
    :type device: device
    :return: Tensors with cuda attribute set to GPU (if available).
    :rtype: Tensor
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """
    Thin wrapper for moving tensors in a DataLoader to a chosen device.
    
    *Note: getting a little lazy with the comment here, TO DO later.
    """
    def __init__(self, dl: DataLoader, device: device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """
        Yield a batch of data after moving it to device.
        """
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """
        Return number of batches.
        """
        return len(self.dl)

    
####################################################################################################
# Data Loading Pipeline

def _data_loading(data: TextDataset, device: device,
                  labels: Series=None,
                  batch_size: int=128, 
                  shuffle: bool=False,
                  sampler: bool=False,
                  num_workers: int=0,
                  pin_memory: bool=False,
                  drop_last: bool=True) -> DeviceDataLoader:
    """
    Data loading step during model training. This step includes quantisation of 
    text data as well as moving the data to a chosen processing device (CPU or GPU).
    
    :param data: the cleaned and quantised text dataset with labels.
    :type data: TextDataset
    :param labels: the labels in the dataset. Only used for creating the sampler
    :type labels: Series
    :param device: Device to which the input data must be moved. GPU or CPU.
    :type device: device
    :param batch_size: how many samples per batch to load.
    :type batch_size: int
    :param shuffle: set to True to have the data reshuffled at every epoch.
    :type shuffle: bool
    :param sampler: set to True to apply random weighted sampling during batch loading.
    :type sampler: bool
    :param num_workers: how many subprocesses to use for data loading. 
        0 means that the data will be loaded in the main process
    :type num_workers: int
    :pin_memory: the data loader will copy Tensors into CUDA pinned memory before 
        returning them.
    :type pin_memory: bool
    :param drop_last: set to True to drop the last incomplete batch, if the 
        dataset size is not divisible by the batch size. If False and the size of 
        dataset is not divisible by the batch size, then the last batch will be 
        smaller.
    :type drop_last: bool
    """
    # Apply sampler if specified
    if sampler:
        sampler = get_sampler(labels)
        shuffle = False
    else:
        sampler = None
        shuffle = True
        
    # Load dataset into dataloader
    dl = DataLoader(data, batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    num_workers=num_workers,
                    drop_last=drop_last, 
                    pin_memory=pin_memory)
    
    # Move data to the selected processing device.
    device_dl = DeviceDataLoader(dl, device)
    return device_dl


def data_loading(train_df: DataFrame, test_df: DataFrame,
                 text_col: str='comment', label_col:str='rating',
                 alphabet: str="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}\n",
                 max_length: int=1014,
                 **params) -> Tuple[DeviceDataLoader]:
    """
    Convert raw train and test data into torch compatible format. The steps taken 
    includes text quantisation and moving data to a selected device. The selection 
    of processing device will always prefer a GPU if a CUDA enabled GPU is available.
    
    :param train_df: Input training data consisting of texts and labels.
    :type train_df: DataFrame
    :param test_df: Input testing data consisting of texts and labels.
    :type test_df: DataFrame
    :param text_col: the name for the text column in the data.
    :type text_col: str
    :param label_col: the name for the label column in the data.
    :type label_col: str
    :param alphabet: the alphabet to be used for text quantisation. It is critical to escape 
        special characters.
    :type alphabet: str
    :param max_length: the maximum text input length used for either truncating or padding 
        of the quantised text matrix.
    :type max_length: int
    :param **params: key word arguments. For further details refer to function parameters of 
        _data_loading function.
    :type **params: dict 
    """
    # Initialise Quantisation parameters for TextDataset
    TextDataset.alphabet = alphabet
    TextDataset.max_length = max_length
    
    #Convert raw input data to quantised tensors
    train_texts, train_labels = train_df[text_col], train_df[label_col]
    test_texts, test_labels = test_df[text_col], test_df[label_col]
    train_set = TextDataset(train_texts, train_labels)
    test_set = TextDataset(test_texts, test_labels)
    
    #Get availabel processing device
    device = get_default_device()
    
    #Move data to selected device and create data loaders
    train_dl = _data_loading(train_set, device, train_df[label_col], **params['train_params'])
    test_dl = _data_loading(test_set, device, None, **params['test_params'])
    
    return train_dl, test_dl


####################################################################################################
# Main

if __name__ == "__main__":
    pass