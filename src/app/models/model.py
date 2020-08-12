"""
Author: Mason Hu
------------------------------------------------------------------

This script contains the internal representation for the model
resource.

------------------------------------------------------------------
"""

# Standard imports
import os
from typing import Tuple

# 3rd party imports
from torch import cuda, load, Tensor, max as t_max
import torch.nn.functional as F

# Project level imports
from models.ai.ai_model import CharCNN
from models.ai.preprocessing import preprocess_text
from models.ai.data_loader import TextDataset


########################################################################################################################
class ModelModel():

    def __init__(self, model_name: str = 'model_trustpilot.pth', model_path: str = './models/ai'):

        # Load the model to GPU if available
        if cuda.is_available():
            trained_weights = load(os.path.join(model_path, model_name))
        else:
            trained_weights = load(os.path.join(model_path, model_name), map_location="cpu")
        self.model = CharCNN()  # Model parameters are currently fixed, this may need to change
        self.model.load_state_dict(trained_weights)
        self.model.eval()
        self.pred_str = ""
        self.score = ""
        self.pred = None

    @classmethod
    def _predict(cls, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Take in a Tensor of multiclass vectors, convert these vectors to probability
        vectors, then pick the class with maximum probability and return its index
        as the prediction.

        :param x: Output vector from a forward pass of a model, usually have dimension
            (batch_size x num_classes).
        :type x: Tensor
        :return: Class predictions corresponding to max probability
        :rtype: Tensor
        """
        # use output of the forward pass of a model to make predictions
        # Pytorch do not accept single samples, so have to add a fake batch dimension
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        # Convert output to probabilities with Softmax
        out_probs = F.softmax(x, dim=1)
        # Based on the probabilities, choose most likely class for prediction
        prob, pred = t_max(out_probs, dim=1)
        return prob.item(), pred.item()

    def predict(self, input: str) -> str:
        """
        Simple prediction function that takes in a string input, and makes
        a prediction on its sentiment using the loaded model.
        """
        out_classes = {0: "Poor", 1: "Average", 2: "Good"}
        input_prep = preprocess_text(input, infer=True)
        input_quant = TextDataset.quant_text(input_prep)
        input_quant = Tensor(input_quant.to_numpy()).unsqueeze(0)  # give it batch dimension
        prob, pred = ModelModel._predict(self.model(input_quant))

        # Probability can be negative for 3 classes output, readjust based on prediciton index
        if pred == 0:
            prob = (0.33 - 0) * (1 - prob) + 0

        elif pred == 1:
            prob = (0.67 - 0.33) * prob + 0.33

        elif pred == 2:
            prob = (1 - 0.67) * prob + 0.67

        self.pred_str = out_classes[pred]
        self.pred = pred
        self.score = prob
        return self.pred_str, self.pred, self.score

    def json(self):
        return {"prediction": self.pred_str,
                "suggested_rating": self.pred,
                "sentiment_score": float(self.score)}


if __name__ == "__main__":
    model = ModelModel()
    review = "This is meh"
    print(model.predict(review))
