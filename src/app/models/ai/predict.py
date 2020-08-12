"""
Author: Mason Hu
------------------------------------------------------------------

This is the script for making sentiment inferences using a trained 
CharCNN.

------------------------------------------------------------------
"""

# Standard lib imports
import json
import os

# 3rd party imports
from torch import load, device, Tensor

# Project level imports
from ai_model import CharCNN
from src.preprocessing import preprocessing
from src.data_loader import TextDataset
from src.pipeline import predict

import pdb

##################################################################
# Load config file for dir reference

fp = os.path.join
with open("config.json", 'r') as conf_file:
    config = json.load(conf_file)
    config['root_dir'] = os.getcwd()
    for k, v in config.items():
        if k != 'root_dir':
            config[k] = fp(config['root_dir'], fp(*v))


##################################################################
# Prediction functions

def load_model(model_path: str=config['model_path']) -> CharCNN:
    """
    Loads the best model (wrt accuracy) in the models directory.
    If no model exist in the directory, return error.
    """
    # Check if the model directory is empty
    if len(os.listdir(model_path)) == 0:
        print("No model found in directory!")
        return None
    else:
        models = [x for x in os.listdir(model_path) if '.pth' in x]
        best_model = ''
        best_f1 = 0
        for model in models:
            f1_score = float(model.split("_")[-1].replace(".pth", ""))
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model
    # This need to be changed if the model structure is different
    model = CharCNN(1, 70, 1014, 3)
    best_model = fp(model_path, best_model)
    model.load_state_dict(load(best_model, map_location=device('cpu')))
    return model


def infer(input: str, model: CharCNN) -> str:
    """
    Simple prediction function that takes in a string input, and makes
    a prediction on its sentiment using an existing model.
    """
    out_classes = {0:"Poor", 1:"Average", 2:"Good"}    
    input_prep = preprocessing(input, infer=True)
    input_quant = TextDataset.quant_text(input_prep)
    input_quant = Tensor(input_quant.to_numpy()).unsqueeze(0) # give it batch dimension
    prediction = predict(model(input_quant)).item()
    output = out_classes[prediction]
    return f"Predicted Sentiment: {output}\n"


##################################################################
# Main

if __name__ == "__main__":
    while True:
        input_str = input("Please provide a input sentence: ")
        model = load_model()
        print(infer(input_str, model))
        next_str = input("Make another prediction? [Y/N]\n")
        if next_str.lower() not in ('y', 'n'):
            print("Did not enter a valid option!")
            next_str = input("Make another prediction? [Y/N]\n")
        elif next_str == 'n':
            break