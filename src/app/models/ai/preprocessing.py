"""
Author: Mason Hu
------------------------------------------------------------------

This script contains all relevant functions for data preprocessing 
required prior to model training.

------------------------------------------------------------------
"""

# Standard lib imports
import re
from typing import Union, Generator

# 3rd party imports
from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader

# Project level imports


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
# Main Test for preprocessing

if __name__ == '__main__':
    pass