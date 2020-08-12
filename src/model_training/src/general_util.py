"""
Author: Mason Hu
------------------------------------------------------------------

This script contains uitility functions for model training.

------------------------------------------------------------------
"""

# Standard lib imports
from typing import Union, Generator, Tuple, List
from collections import Counter
from math import ceil

# 3rd party imports
from pandas import DataFrame, Series, read_csv, concat
from pandas.io.parsers import TextFileReader
from numpy import bincount
from matplotlib.ticker import FuncFormatter
from torch import Tensor, cuda
from torch.utils.data import WeightedRandomSampler

# Project level imports

#######################################################################################

def check_basic_data_stats(data: TextFileReader):
    """
    Take in a pandas.io.parsers.TextFileReader object and check its total memory usage 
    per column in the data.
    """
    len_data = 0
    memory_usage = Series(dtype='object')
    
    for chunk in data:
        # Determine length of dataset
        len_data += len(chunk)
    
        # Determine memory usage of dataframe
        if len(memory_usage) == 0:
            memory_usage = chunk.memory_usage(deep=True)/1e6
        else:
            memory_usage += chunk.memory_usage(deep=True)/1e6

    print(f'Number of reviews scraped from Trustpilot: {len_data:,} reviews\n')
    print(f'Memory usage per column (MB):\n{memory_usage}\n')
    print(f'Total memory usage of dataframe: {memory_usage.sum()/1e3:.2f} GB\n')
    return chunk # this is the last chunk in the entire data set, return for display


def label_class_fcount(data: Union[DataFrame, Generator[DataFrame, None, None]], 
                       label_col: str='rating') -> DataFrame:
    """
    Take either a dataframe or a generator of dataframes and return a Series representing 
    the frequency count of existing label classes. In addition, also calculate the corresponding 
    probability of each class.
    
    :param data: the data containing the label column for which class frequency count 
                 must be determined.
    :type data: Union[DataFrame, Generator[DataFrame, None, None]]
    :param label_col: The coloumn in the data for which frequency count must be determined.
                      Defaults to 'rating' corresponding ot the label column in the original data.
    :type label_col: str
    :raise: TypeError if the input data format isn't either a DataFrame or Generator[DataFrame, None, None].
    :return: frequency count of all distinct classes within the label data as well as their corresponding probabilities.
    :rtype: DataFrame
    """
    data = deepcopy(data)
    if isinstance(data, DataFrame):
        fcount = data[label_col].value_counts()
        sorted_idx = fcount.index.sort_values()
        fcount = fcount[sorted_idx]
        prob = fcount / fcount.sum()
        return DataFrame({'frequency': fcount, 'probability': prob})
    
    elif isinstance(data, Generator):
        fcount_total = None
        for chunk in data:
            if fcount_total is None:
                fcount_total = chunk[label_col].value_counts()
            else:
                fcount_total += chunk[label_col].value_counts()
        sorted_idx = fcount_total.index.sort_values()
        fcount_total = fcount_total[sorted_idx]
        prob = fcount_total / fcount_total.sum()
        return DataFrame({'frequency': fcount_total, 'probability': prob})
    
    else:
        raise TypeError("""Allowed types are one of the following: 
            - pandas.DataFrame
            - Generator[pandas.DataFrame, None, None]""")
        
        
def plot_label_dist(label_dist: DataFrame, plot_type: str='frequency') -> None:
    """
    Helper function to plot the label class distribution.
    
    :param label_dist: the label class distribution summary.
    :type label_dist: DataFrame
    :param plot_type: the type of plot to show. Can be 'frequency' or 'probability'.
    :type plot_type: str
    :return: Nothing
    :rtype: None
    """
    ax = label_dist[plot_type].plot.bar(figsize=(10,5))
    ax.grid()
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(x, ',.2f'))) # Set y axis 1000s separator
    ax.tick_params(axis='x', labelrotation=0) 
    ax.set_xlabel('Label Classes')
    ax.set_ylabel(plot_type)
    ax.set_title(f"Label class {plot_type}")
    

def get_class_weights(labels: Series) -> Tensor:
    """
    Calculate class weightings based on each class' proportion
    in the label.
    
    :param labels: The labels in the training dataset.
    :type labels: Series
    :return: A tensor of weights.
    :rtype: Tensor
    """
    # Calculate class weightings
    class_counts = dict(Counter(labels))
    m = max(class_counts.values())
    for c in class_counts:
        class_counts[c] = m / class_counts[c]
    # Convert weightings to tensor
    weights = []
    for k in sorted(class_counts.keys()):
        weights.append(class_counts[k])
    weights = Tensor(weights)
    # Move weights to GPU if available
    if cuda.is_available():
        weights = weights.cuda()
    return weights


def stratified_shuffle(df: DataFrame, batch_size: int, 
                       label_col: str, random_state: int) -> DataFrame:
    """
    Shuffle the input dataset by batches as specified by batch_size. The
    result dataset will contain the same number of samples as the original 
    dataset, however, each sub portion of the result dataset (with number 
    of samples equal to batch_size) will have the same class proportions 
    as per the original dataset.
    
    e.g.
    Original dataset: 
    class 1: 0.2
    class 2: 0.6
    class 3: 0.2
    
    Result dataset:
    batch 1:
        class 1: 0.2
        class 2: 0.6
        class 3: 0.2
    batch 2:
        class 1: 0.2
        class 2: 0.6
        class 3: 0.2
        .
        .
        .
    
    :param df: Input dataframe
    :type df: DataFrame
    :param batch_size: The size of each stratified batch within the result dataset
    :type batch_size: int
    :param label_col: the name of the label column in the dataset.
    :type label_col: str
    :param random_state: see of the random sampling of each stratified batch. Set 
        for reproducible results.
    :type random_state: int
    """
    df_len = df.shape[0]
    # Calculate the proportion of each class in the original dataset
    f_counts = df[label_col].value_counts()
    proportions = df[label_col].value_counts(normalize=True).round(3)
    # Calculate the sub batch size for each class per batch based on class proportions in the original dataset
    batch_sizes = (proportions * batch_size).apply(lambda x: int(x))
    diff = batch_size - batch_sizes.sum()
    batch_sizes[2] = batch_sizes[2] + diff # Add this to cater for rounding errors to assert batch_size
    # Calculate the number of iterations to loop through, neglect the odd dataset that can't fit
    num_iters = int((f_counts / batch_sizes).min())
    out_df = DataFrame()
    # Loop through the entire dataset by batch_size
    for i in range(num_iters):
        # For each class, randomly extract the calculated number of samples per class
        # The extracted samples are dropped from the original dataset to avoid duplicate samples in the next iteration
        for idx, size in batch_sizes.iteritems():
            temp_df = df[df[label_col]==idx]
            batch = temp_df.sample(n=size, random_state=random_state)
            batch_idx = batch.index
            df = df.drop(index=batch_idx).reset_index(drop=True)
            out_df = out_df.append(batch)
            
    return out_df.reset_index(drop=True)


def get_sampler(labels: Series):
    """
    Define a weighted random sampler (geometric distribution), to be used 
    with the DataLoader object from torch. This should be applied to the 
    training set which will replace the random shuffling during batch loading. 
    The results is that each batch loaded should contain very similar class 
    distribution across all existing classes in the training data. This should
    reduce overfitting due to class imbalance.
    
    :param labels: The labels from the dataset.
    :type labels: Series
    :return: a sampler
    :rtype: WeightedRandomSampler
    """
    # Get count per class in the dataset
    classcount = bincount(labels).tolist()
    # Calculate weighting of each class
    class_weights = 1 / Tensor(classcount)
    # Assign class weighting to each sample in dataset
    sample_weights = class_weights[labels]
    if cuda.is_available():
        sample_weights.cuda()
    # Apply calculated sample weight to the sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler
    

class AverageMeter():
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        
#######################################################################################
# Main

if __name__ == '__main__':
    pass