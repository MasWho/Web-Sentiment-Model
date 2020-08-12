"""
Author: Mason Hu
------------------------------------------------------------------

This script contains all relevant functions and classes for data loading 
functionalities.

------------------------------------------------------------------
"""

# Standard lib imports
from typing import Union

# 3rd party imports
from pandas import DataFrame, Series, concat
from numpy import identity, zeros
from torch import Tensor
from torch.utils.data import Dataset

# Project level imports


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
                    text_mat = concat([text_mat, df_to_append], axis=0).reset_index(drop=True)  # text_mat dimension: 1014 x 70
                        
            return text_mat  # text_mat dimension: 1014 x 70

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
# Main

if __name__ == "__main__":
    pass