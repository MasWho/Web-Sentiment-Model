"""
Author: Mason Hu
------------------------------------------------------------------

This script contains all relevant functions for data preprocessing 
required prior to model training.

------------------------------------------------------------------
"""

# Standard lib imports
import re
from typing import Union, Iterator, Iterable, Generator, Tuple, List
from copy import deepcopy
from os.path import isfile, isdir, join as fp

# 3rd party imports
from pandas import DataFrame, Series, read_csv, concat
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Project level imports
from src.data_loader import TextDataset


########################################################################################################
# Cleaning / Preprocessing data

def remove_emoji(text: str) -> str:
    """
    Remove emojis from a input string of text.
    
    :param text: text to be processed.
    :type text: str
    :returns: processed text with emojis removed.
    :rtype: str
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text).strip()


def remove_hashtags(text: str) -> str:
    """
    Remove hashtags from the input text. e.g. #happy
    
    :param text: text to be processed.
    :type text: str
    :returns: processed text with hashtags removed.
    :rtype: str
    """
    clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
    return clean_text.strip()


def remove_user_mentions(text: str) -> str:
    """
    Remove user mentions from the input text. e.g. @mason
    
    :param text: text to be processed.
    :type text: str
    :returns: processed text with user mentions removed.
    :rtype: str
    """
    clean_text = re.sub(r'@[A-Za-z0-9_]+', "", text)
    return clean_text.strip()


def remove_urls(text: str) -> str:
    """
    Remove ulrs from the input text. e.g. https:\\www.google.com
    
    :param text: text to be processed.
    :type text: str
    :returns: processed text with urls removed.
    :rtype: str
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    clean_text = re.sub(r'http\S+', '', text) # urls can span across many lines
    return clean_text.strip()


def lower(text) -> str:
    """
    Convert input string to lower case.
    
    :param text: text to be processed.
    :type text: str
    :returns: processed text converted to lower case.
    :rtype: str
    """
    return text.lower()


# Not needed for inference
def clean_label(batch: DataFrame, col: str) -> DataFrame:
    """
    Take an input dataframe (probably part of a batch process) and make sure that 
    the label columns contain all integers in the range of 1 - 5.
    
    :param batch: a dataframe.
    :type batch: DataFrame
    :param col: the label column.
    :type col: str
    :returns: a processed dataframe with the label column cleaned.
    :rtype: DataFrame
    """
    # First check if the label column has dtype of int
    int_check = batch[col].dtype == 'int'
    # If the column is already in integer format
    # only check if values are within range
    if int_check:
        batch = batch[batch[col].isin(range(1,6))]
    # If the column isn't in integer format
    # Filter out rows where label is not a number
    # Then check if values are within range
    else:
        num_rows = batch.shape[0]
        batch = batch[batch[col].str.isnumeric()]
        batch = batch.astype('int')
        batch = batch[batch[col].isin(range(1,6))]
        print(f"Found bad label: removed {batch.shape[0]-num_rows} rows.")
    return batch


def preprocess_text(data: Union[str, DataFrame, Series, TextFileReader], 
                    text_col: str='comment', 
                    label_col: str='rating', 
                    infer: bool=False) -> Union[str, DataFrame, Generator[DataFrame, None, None]]:
    """
    Applies preprocessing to input data. Allowed data types can be a input string of text, 
    a pandas DataFrame or a pandas TextFileReader. The latter input type generally occrurs 
    if the dataset is large and won't fit in memory as a whole. In that case, the returned 
    data is a generator. This can be thought of as chaining generators to produce a stream 
    of data that will fit in memory.
    
    :param data: The input text data to be preprocessed.
    :type data: Union[str, DataFrame, TextFileReader]
    :param text_col: The column of dataframe that contains the text data. If input is a 
                     single string, this parameter is ignored. Defaults to 'comment'
                     corresponding to the original dataset column name.
    :type text_col: str
    :param label_col: The column of dataframe that contains the label data. If input is a 
                      single string, this parameter is ignored. Defaults to 'rating'
                      corresponding to the original dataset column name.
    :type label_col: str
    :param infer: parameter indicating if inference is called.
    :type infer: bool
    :raise: TypeError if the input data format isn't either a str, DataFrame or TextFileReader.
    :returns: Preprocessed text data.
    :rtypes: Union[str, DataFrame, Generator[DataFrame, None, None]] 
    """
    preprocessing_steps = {
        'Removing emojis': remove_emoji,
        'Removing hashtags': remove_hashtags,
        'Removing user mentions': remove_user_mentions,
        'Removing urls': remove_urls,
        'Converting to lower case': lower
    }
    
    def preprocess_text_chunk(chunk: DataFrame) -> DataFrame:
        # Apply the preprocessing pipeline to a dataframe
        for step, func in preprocessing_steps.items():
            chunk[text_col] = chunk[text_col].apply(func)
        chunk = chunk.dropna()
        # Only clean labels if it's not a inference call
        if not infer:
            chunk = clean_label(chunk, label_col)
        return chunk
    
    # If input is a single instance of text, then we are making inference
    if isinstance(data, str):
        for step, func in preprocessing_steps.items():
            data = func(data)
        return data
    
    # If input is a Series, clean the Series and return it
    elif isinstance(data, Series):
        for step, func in preprocessing_steps.items():
            data = data.apply(func)
        data = data.dropna()
        return data
    
    # If input is a dataframe, clean the dataframe and return it
    elif isinstance(data, DataFrame):
        return preprocess_text_chunk(data)

    # If input is a TextFileReader, 
    # clean it but return a generator to improve memory efficiency
    elif isinstance(data, TextFileReader):
        return (preprocess_text_chunk(chunk) for chunk in data)
    
    else:
        raise TypeError("""Allowed types are one of the following:
            - str, 
            - pandas.DataFrame
            - pandas.io.parsers.TextFileReader""")
        
########################################################################################################
# Preparing data for training (This is not needed for inference)

def group_labels(data: Union[DataFrame, Generator[DataFrame, None, None]], 
                 label_col: str='rating') -> Union[DataFrame, Generator[DataFrame, None, None]]:
    """
    Assign new groups to old label classes.
    """
    def _assign_class(data: DataFrame):
        # Assign group of labels to a new set of classes 0,1,2 representing poor, average and bad
        data.loc[(data[label_col]==1) | (data[label_col]==2), label_col] = 0 # Group [1,2] to 0
        data.loc[(data[label_col]==3), label_col] = 1 # Group [3] to 1
        data.loc[(data[label_col]==4) | (data[label_col]==5), label_col] = 2 # Group [4,5] to 2
        return data
    
    if isinstance(data, DataFrame):
        return _assign_class(data)
    
    elif isinstance(data, Generator):
        return (_assign_class(chunk) for chunk in data)
    
    else:
        raise TypeError("""Allowed types are one of the following: 
            - pandas.DataFrame
            - Generator[pandas.DataFrame, None, None]""")
        

def raw_train_test_split(data: Union[DataFrame, Generator[DataFrame, None, None]], 
                         train_size: float=0.9, random_state: int=None,
                         text_col: str='comment', label_col: str='rating',
                         class_data_usage: List[float]=[1,1,0.16],
                         classes: List[int]=[0,1,2]) -> Union[Tuple[DataFrame, DataFrame], 
                                                              Tuple[Generator[DataFrame, None, None], 
                                                                    Generator[DataFrame, None, None]]]:
    """
    Take a dataset and split it into training and test set. Flexibility is provided that the user may 
    provide the possible label classes that exist in the dataset, and specify the percentage data usage 
    on a per class basis. For example, given a data set that contains a binary label with distinct values 
    [1, 2], the user can specify the percentage of data to be used per class for constructing the training 
    and test set. If in this scenario the user would like to use 50% and 80% of data corresponding to the two 
    label classes, the user may pass a list of proportions e.g [0.5, 0.8] to the class_data_usage parameter.
    
    :param data: the input dataset to be split into train and test set.
    :type data: Union[DataFrame, Generator[DataFrame, None, None]]
    :param train_size: proportion for the training set in the range between 0 and 1. Default to 0.9.
    :type train_size: float
    :param random_state: if a value is passed to this parameter, the random seed will be fixed hence resulting 
        in reproducible dataset split.
    :type random_state: int
    :param text_col: the column name of the text.
    :type text_col: str
    :param label_col: the column name of the label.
    :type label_col: str
    :param class_data_usage: a list of usage proportions for the existing sub datasets each corresponding to a 
        label class. Individual values should be in the range of 0-1, and the elements in  
        the list must correlate to the elements in classes list.
    :type class_data_usage: List[float]
    :param classes: a list of label classes that exist in the dataset. The elements in the lsit must correlate 
        to the elements in the class_data_usage list.
    :return: the split train and test set. Note this function allows output as a generator to reduce memory usage.
        The output data type is directly dependent on the input datatype.
    :rtype: Union[Tuple[DataFrame, DataFrame], 
                  Tuple[Generator[DataFrame, None, None], Generator[DataFrame, None, None]]]:
    """
    def _raw_train_test_split(data: Union[DataFrame, Generator[DataFrame, None, None]]):
        train_set = []
        test_set = []
        for class_, usage in zip(classes, class_data_usage):
            class_split = data[data[label_col]==class_]
            class_train, class_test = train_test_split(class_split, train_size=train_size, random_state=random_state)
            class_train = class_train.iloc[:int(usage*class_train.shape[0]), :] # Grab proportion of data based on class usage specified
            class_test = class_test.iloc[:int(usage*class_test.shape[0]), :]
            train_set.append(class_train)
            test_set.append(class_test)
        return concat(train_set), concat(test_set)
    
    if isinstance(data, DataFrame):
        train_set, test_set = _raw_train_test_split(data)
        return train_set, test_set
    
    elif isinstance(data, Generator):
        train_test_gen = (_raw_train_test_split(chunk) for chunk in data)
        return train_test_gen
    
    else:
        raise TypeError("""Allowed types are one of the following: 
            - pandas.DataFrame
            - Generator[pandas.DataFrame, None, None]""")
        
        
def save_train_test_data(data: Generator[DataFrame, None, None], 
                         train_path: str, test_path: str) -> Tuple[DataFrame, DataFrame]:
    """
    Save the train and test datasets to their respective folders. This function is 
    specifically written to save generator data to csv files.

    :param data: the generator that will return train and test sets.
    :type data: Generator[DataFrame, None, None]]
    :param train_path: path to which the training set must be saved.
    :type train_path: str
    :param test_path: path to which the test set must be saved.
    :type test_path: str
    :return: A tuple containing the training set and testing set.
    :rtype: Tuple[DataFrame, DataFrame]
    """
    final_train_set = []
    final_test_set = []
    for train_set, test_set in tqdm(data, total=60):
        final_train_set.append(train_set)
        final_test_set.append(test_set)
    final_train_set = concat(final_train_set)
    final_test_set = concat(final_test_set)
    final_train_set.to_csv(train_path, index=False)
    final_test_set.to_csv(test_path, index=False)
    print(f"\nSuccessfully prepared the data!!!\n", 
          f" - Train data saved in: {train_path}\n", 
          f" - Test data saved in: {test_path}\n")
    return final_train_set, final_test_set


def _train_preprocessing(raw_path, train_path, test_path,
                         infer: bool=False,
                         chunksize: int=100_000,  
                         encoding: str='utf-8', 
                         text_col: str='comment', 
                         label_col: str='rating',
                         train_size: float=0.9,
                         random_state: int=None,
                         class_data_usage: List[float]=[1,1,0.16], 
                         classes: List[int]=[0,1,2]) -> Tuple[DataFrame, DataFrame]:
    """
    Collects all preprocessing functions together for training the model. This differs 
    to the preprocessing steps during testing and / or inference. There is no need to 
    group the data by label classes and stratify during testing / inference.
    
    *Note: This is really a convenience function since this function is wrapped by 
    preprocessing(). Keyword arguments are passed down by means of unpacking, which 
    allows for simpler future changes in function parameters.
    
    :param raw_path: path to which the raw unprocessed data is located.
    :type raw_path: str
    :param train_path: path to which the raw unprocessed data is located.
    :type train_path: str
    :param test_path: path to which the raw unprocessed data is located.
    :type test_path: str
    :param infer: parameter indicating if inference is called.
    :type infer: bool
    :param chunk_size: this parameter dictates the chunk size during batch data reading.
    :type chunk_size: int
    :param encoding: the raw data encoding.
    :type encoding: str
    :param text_col: the name for the text column in the data.
    :type text_col: str
    :param label_col: the name for the label column in the data.
    :type label_col: str
    :param train_size: the training set size (0-1) to be applied for splitting the processed data.
    :type train_size: float
    :param random_state: if a value is passed to this parameter, the random seed will be fixed hence resulting 
        in reproducible dataset split.
    :type random_state: int
    :param class_data_usage: a list of usage proportions for the existing sub datasets each 
        corresponding to a label class. Individual values should be in the range of 0-1, and the 
        elements in the list must correlate to the elements in classes list.
    :type class_data_usage: List[float]
    :param classes: a list of label classes that exist in the dataset. The elements in the lsit 
        must correlate to the elements in the class_data_usage list.
    :type classes: List[int]
    :return: processed train and test set (before quantisation)
    :rtype: Tuple[DataFrame, DataFrame]
    """
    df = read_csv(raw_path, chunksize=chunksize, usecols=[text_col, label_col], encoding=encoding)
    clean_df = preprocess_text(df, text_col, label_col, infer=infer)
    clean_df = group_labels(clean_df, label_col)
    clean_train_test_df = raw_train_test_split(clean_df, text_col=text_col, 
                                               label_col=label_col, train_size=train_size,
                                               random_state=random_state,
                                               class_data_usage=class_data_usage, 
                                               classes=classes)
    df_train, df_test = save_train_test_data(clean_train_test_df, train_path, test_path)
    return df_train, df_test

    
def preprocessing(data: Union[str, Series, DataFrame]=None, 
                  raw_path: str=None, 
                  train_path: str=None, 
                  test_path: str=None, 
                  infer: bool=False, 
                  **kwargs) -> Union[str, Series, Tuple[DataFrame, DataFrame]]:
    """
    Wrapper around _train_preprocessing. Checks if train and test data have already been
    saved, and load them directly if the saved data exist. Furthermore, if the call to 
    the function is for inference purposes, the apply preprocess_text directly to the text 
    input and skip the grouping labels step.
    
    :param data: Input text data during inference. If it's not inference this parameter is 
        ignored.
    :type data: Union[str, Series, DataFrame]
    :param raw_path: path to which the raw unprocessed data is located.
    :type raw_path: str
    :param train_path: path to which the raw unprocessed data is located.
    :type train_path: str
    :param test_path: path to which the raw unprocessed data is located.
    :type test_path: str
    :param infer: parameter indicating if inference is called.
    :type infer: bool
    :param **kwargs: key word arguments. For further details refer to function parameters of 
        _train_preprocessing function.
    :type **kwargs: dict
    :return: train and test set as dataframes if it's for a training process. Otherwise return 
        only processed text as  output.
    :rtype: Union[str, Series, Tuple[DataFrame, DataFrame]]
    """
    # If it's a inference call, then only operate on text since there won't be any labels
    if infer:
        return preprocess_text(data, infer=True)
    
    else:
        # Check to see if train and test set exist, if so, then read in the csv files directly
        # Skip the preprocessing steps.
        if isfile(train_path) and isfile(test_path):
            print("\nTraining and testing data have already been prepared:\n", 
                  f" - Train set is located in: {train_path}\n", 
                  f" - Test set is located in: {test_path}\n")
            
            df_train = read_csv(train_path).dropna().reset_index(drop=True)
            df_test = read_csv(test_path).dropna().reset_index(drop=True)
            
            print("The label distribution for training set:\n",
                   df_train[kwargs['label_col']].value_counts()/df_train.shape[0])
            print("The label distribution for testing set:\n",
                   df_test[kwargs['label_col']].value_counts()/df_test.shape[0], "\n")
            
            return df_train, df_test
        # If the train and test set don't exist, then go the whole 9 yards
        # In the process, also saves the train set and test set as csv files
        else:
            print("\n\n***** Preparing data for training *****")
            df_train, df_test = _train_preprocessing(raw_path, train_path, test_path, **kwargs)
            
            df_train = read_csv(train_path).dropna().reset_index(drop=True)
            df_test = read_csv(test_path).dropna().reset_index(drop=True)
            
            print("The label distribution for training set:\n",
                   df_train[kwargs['label_col']].value_counts()/df_train.shape[0])
            print("The label distribution for testing set:\n",
                   df_test[kwargs['label_col']].value_counts()/df_test.shape[0], "\n")
            
            return df_train, df_test


########################################################################################################
# Main Test for preprocessing

if __name__ == '__main__':
    # Define relevant paths
    data_path = './data'
    raw_path = fp(data_path, 'raw', 'scraped_data.csv')
    train_path = fp(data_path, 'train', 'train_clean.csv')
    test_path = fp(data_path, 'test', 'test_clean.csv')
    
    df_train, df_test = preprocessing(raw_path=raw_path, train_path=train_path, 
                                      test_path=test_path, infer = False,
                                      chunksize=100000, text_col='comment', 
                                      label_col='rating', train_size=0.9,
                                      randome_state = 10,
                                      classes=[0,1,2], class_data_usage=[1,1,0.16])