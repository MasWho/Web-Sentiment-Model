"""
Author: Mason Hu
------------------------------------------------------------------

This script contains the implementation of a CharCNN using PyTorch.

------------------------------------------------------------------
"""

# Standard lib imports
import numpy as np
from typing import Tuple

# 3rd party lib imports
from torch import rand, Tensor, nn, cuda

# project level imports


################################################################################################
# CharCNN implementation

class CharCNN(nn.Module):
    """
    Character level CNN implementation as per the paper authored by Zhang et.al. The paper 
    is titled "Character-level convolutional Networks for Text Classification - 2016". The 
    model consists of 9 layers including 6 convolutional layers and 3 fully connected layers.
    Details are shown below.
                
                Conv Layers
    -------------------------------------------------
    Layer    Features    Kernel    Pool    Activation
    -----    --------    ------    ----    ----------
      1         256         7        3        ReLU
      2         256         7        3        ReLU
      3         256         3        N/A      ReLU 
      4         256         3        N/A      ReLU
      5         256         3        N/A      ReLU
      6         256         3        3        ReLU
      
                FC Layers
    -----------------------------------
    Layer    Features    Dropout    Activation
    -----    --------    -------    ----------
      7        1024        p=0.5       ReLU
      8        1024        p=0.5       ReLU
      9        TBC         N/A         N/A
      
    Furthermore, the paper initialised the model weights using a Gaussian distribution with 
    a standard deviation of 0.05. Note there were no specific details regarding padding of 
    the convolutional layer inputs, and 'SAME' padding is assumed.
    """
    
    def __init__(self, batch_size: int, alph_len :int, max_len: int, num_classes: int):
        """
        CharCNN model constructor. During model instantiation, the trainable model weights 
        are initialised using Gaussian distribution.
        
        :param batch_size: Number of input samples.
        :type batch_size: int
        :param alph_len: The number of characters in the alphabet for text quantisation. This 
            dictates the dimension of the first convolutional layer.
        :type alph_len: int
        :param max_len: The assume maximum input text length. This is the input features to 
            the model, and dictates the input feature length for the first linear layer.
        :type max_len: int
        :param num_classes: The number of classes for prediction. This dictates the number of 
            output units of the last linear layer.
        :type num_classes: int
        :return: Nothing.
        :rtype: None
        """
        
        super(CharCNN, self).__init__()
    
        # Convolutional layer architecture
        self.conv_layers = nn.Sequential(
            # conv1 -> (b x 256 x 338): 
            nn.Conv1d(alph_len, 256, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3),
            # conv2 -> (b x 256 x 112)
            nn.Conv1d(256, 256, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3),
            # conv3 -> (b x 256 x 112)
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            # conv4 -> (b x 256 x 112)
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            # conv5 -> (b x 256 x 112)
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            # conv6 -> (b x 256 x 37) -> 9472 features
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )
        
        # Determine linear layer input shape
        input_shape = (batch_size, alph_len, max_len)
        self.lin_input_features = self._get_lin_features(input_shape)
        
        # Linear layer architecture
        self.linear_layers = nn.Sequential(
            # linear7
            nn.Linear(self.lin_input_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            # linear8
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            # linear9
            nn.Linear(1024, num_classes),
        )
        
        # Calculate total number of trainable parameters in the model
        self.total_params = self._get_total_params()
        # Random initialisation of weights using gaussian distribution
        self.apply(self._init_weights)
        # If GPU is available, instantiate the model on GPU
        if cuda.is_available():
            self.cuda()
    
    def _get_lin_features(self, shape: Tuple[int]) -> int:
        """
        Convenience function to calculate the input feature length for 
        the first linear layer in the model.
        
        :param shape: Input shape to the model in the form of (b x w x l) where 
            b, l and w are the input batch size, width (number of rows) and 
            length (number of columns) respectively.
        :type shape: Tuple[int]
        :return: The number of input features to the first linear layer in the 
            model.
        :rtype: int
        """
        x = rand(shape)
        x = self.conv_layers(x)
        return x.view(x.size(0), -1).shape[1]
    
    def _get_total_params(self) -> int:
        """
        Convenience function for calculating the total number of trainable parameters 
        in the model.
        """
        total_params = sum([p.numel() for p in self.parameters() 
                            if p.requires_grad])
        return total_params
    
    def _init_weights(self, module: nn.Module, mean: float=0., std: float=0.05) -> None:
        """
        Convenience function for initialising the weights for a single module in the 
        model using Gaussian distribution with specified mean and standard deviation. 
        This function should be passed to nn.Module.apply which recursively applies
        input function to all sub modules within the model.
        
        :param module: a nn.Module object to which the weight initialisation will be 
            applied.
        :type module: nn.Module
        :param mean: Mean for a Gaussian distribution.
        :type mean: float
        :param std: Standard deviation for a Gaussian distribution.
        :type std: float
        :return: Nothing. Modifies the input modules weights and biases inplace.
        :rtype: None.
        """
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean, std) # inplace op with funct_
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """
        Forward propagation function for the model. The output of the model will 
        have dimension (batch_size x number of classes for prediction)
        """
        x = x.transpose(1,2) # transpose sicne input is actually provided as (max_len x alph_len) but we want the other way around
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the output from the convolutional layers, keep batch dimension intact
        x = self.linear_layers(x)
        return x