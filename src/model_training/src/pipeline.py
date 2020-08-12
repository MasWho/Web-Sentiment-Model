"""
Author: Mason Hu
------------------------------------------------------------------

This script contains functions for the model training process.

*TO DO:
 - Comments
 - Implement TensorboardX

------------------------------------------------------------------
"""

# Standard lib imports
from collections import Counter
from typing import Union, List, Tuple
import os

# 3rd party lib imports
from torch import (nn, cuda, optim, Tensor, float32, cat, no_grad, 
                   max as t_max, sum as t_sum, save as t_save) 
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from pandas import read_csv, Series
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import numpy as np

# Project level imports
from src.general_util import AverageMeter, get_class_weights
from src.data_loader import DeviceDataLoader
from src.model import CharCNN


#####################################################################
# General

def _setup_progress(train_dl: DeviceDataLoader):
    """
    Takes a DeviceDataLoader (basically a generator) and determine
    number of iteration will be required per batch. This is used 
    to set up a progress bar for the training.
    
    :param train_dl: Training dataset.
    :type train_dl: DeviceDataLoader
    :return: Progress bar that wraps the data generator.
    """
    iter_per_epoch = len(train_dl)
    progress_bar = tqdm(enumerate(train_dl), total=iter_per_epoch)
    return iter_per_epoch, progress_bar


def predict(x: Tensor) -> Tensor:
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
    _, preds = t_max(out_probs, dim=1)
    return preds


def accuracy(pred: Tensor, label: Tensor):
    # Use accuracy as a metric here
    check = pred == label
    num_correct = t_sum(check).item()
    accuracy = num_correct / len(pred)
    return accuracy


def report_f1_batch(pred_batch: Tensor, label_batch: Tensor, 
                    class_names: List[Union[str, int]]):
    # Need to first convert tensors back to list for use with sklearn
    # Necessary for calculation of f1 score, better way would be do it with PyTorch
    pred_batch = pred_batch.cpu().numpy().tolist()
    label_batch = label_batch.cpu().numpy().tolist()
    score_dict = classification_report(label_batch, pred_batch, output_dict=True)
    f1_dict = {str(class_): score_dict[str(class_)]['f1-score'] 
               for class_ in class_names}
    print('F1 Scores by class: ')
    for class_, f1 in f1_dict.items():
        content = f"{class_}: {f1:.4f}"
        print(content)


def process_batch(model: CharCNN, batch: Tensor, 
                   optimiser: Union[optim.SGD, optim.Adam], 
                   criterion: CrossEntropyLoss, train=False):
    # Forward propagation on batch
    x, y = batch # x:(b x 1014 x 70)   y:(b x 1)
    output = model.forward(x) # output:(b x 3)
    # Get predictions
    pred = predict(output) # pred:(b x 1)
    # Note cross_entropy function includes softmax already, 
    # therefore compute directly using y and not probabilities
    loss = criterion(output, y)
    # Calculate accuracy and f1 score
    acc = accuracy(pred, y)
    # If this is a training batch process, then perform back propagation
    if train:
        # Clear gradient buffer
        optimiser.zero_grad()
        # Back propagation
        loss.backward()
        # Update gradients
        optimiser.step()
        return (pred, y), {"Train loss": loss.data, 
                           "Train acc": acc}
    
    return (pred, y), {"Val loss": loss.data, 
                       "Val acc": acc}


#####################################################################
# Training

def _init_optimisation(model: CharCNN, 
                       optimiser: str='sgd', 
                       class_weights: Tensor=None, 
                       lr: float=0.01, momentum: float=0.9, 
                       schedule_lr: bool=True) -> Tuple[Union[optim.SGD, optim.Adam], 
                                                         nn.CrossEntropyLoss, MultiStepLR]:
    """
    Initialise the optimisation algorithm which selects:
    1. Balanced or unbalanced crossentropy loss function, if unbalanced, class weightings
       are applied during loss optimisation.
    2. Gradient descent algorithm ca be either SGD with momentum, or ADAM.
    3. The user has the option to enable learning rate scheduling for the optimisation 
       algorithm. The scheduler implements learning rate reduction by halving it every 
       three epochs up to the point where this has been applied 10 times.
       
    :param model: A CharCNN model from which parameters will be updated by the optimiser.
    :type model: CharCNN
    :param optimiser: The type of gradient descent algorithm to use. Can be 'sgd' or 'adam'.
    :type optimiser: str
    :param class_weights: The list of weightings to be applied to each class for the crossentropy
        loss function. Note if unbalance_classes is False, this parameter will be ignored.
    :type class_weights: Tensor
    :param lr: learning rate for the gradient descent. If schedule_lr is True, this will be 
        the initial learning rate.
    :type lr: float
    :param momentum: the velocity coefficient gradient descent with momentum
    :type momentum: float
    :param schedule_lr: Indicator to enable learning rate scheduling with the algorithm mentioned 
        in the function description.
    :type schedule_lr: bool
    :return: 
    """
    # Balance or unbalanced loss function
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss() # The cross_entropy function includes softmax calc, 
                                          # and can have weights applied for different classes
    
    # Choose optimiser
    if optimiser == 'sgd':
        optimiser = optim.SGD(model.parameters(), lr=lr, momentum=momentum) # Stochastic gradient descent optimiser, 
                                                                            # can have momentum term passed to it
    elif optimiser == 'adam':
        optimiser = optim.Adam(model.parameters(), lr=lr)
        
    # Create learning rate scheduler
    if schedule_lr:
        # Multiply optimiser learning rate by 0.5 for each milestone epochs specified in steps
        steps = [x*3 for x in range(1, 11)]
        scheduler = MultiStepLR(optimiser, milestones=steps, gamma=0.5)
    else:
        scheduler = None
        
    return optimiser, criterion, scheduler


def _train_epoch(model: CharCNN, train_dl: DeviceDataLoader,
                 optimiser: Union[optim.SGD, optim.Adam],
                 criterion: CrossEntropyLoss,
                 epoch: int,
                 writer: SummaryWriter,
                 class_names: List[Union[str, int]]=[0,1,2],
                 log_file: str=None,
                 log_f1: bool=True,
                 print_every: int=100):
    
    # Initial training setup
    model.train()
    iter_per_epoch, progress_bar = _setup_progress(train_dl)
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pred_list = list()
    label_list = list()
    for iter_, batch in progress_bar:
        pred_label, scores = process_batch(model, batch, optimiser, 
                                           criterion, train=True)
        try:
            for k in scores:
                if 'loss' in k:
                    loss_batch = scores[k]
                elif 'acc' in k:
                    acc_batch = scores[k]
            pred_batch, label_batch = pred_label
            pred_list.append(pred_batch)
            label_list.append(label_batch)

            # Writing batch results to tensorboard
            writer.add_scalar('Train/Batch/Loss', loss_batch, epoch*iter_per_epoch+iter_)
            writer.add_scalar('Train/Batch/Accuracy', acc_batch, epoch*iter_per_epoch+iter_)
            pred_np = pred_batch.cpu().numpy().tolist()
            label_np = label_batch.cpu().numpy().tolist()
            f1_batch = f1_score(pred_np, label_np, average='weighted')
            writer.add_scalar('Train/Batch/f1', f1_batch, epoch*iter_per_epoch+iter_)

            # Log losses and accuracy
            # sum losses and accuracies over all batches in 1 epoch and calculated the average
            losses.update(loss_batch) 
            accuracies.update(acc_batch)

            # Log learning rate
            lr = optimiser.state_dict()["param_groups"][0]["lr"]
            # Print out training results at fixed batch intervals specified by 'print_every'
            if (iter_ % print_every == 0) and (iter_ > 0):
                print(f"\n[Training - Epoch: {epoch+1}] , LR: {lr:.6f} , ", end='')
                print(f"Iteration: {iter_}/{iter_per_epoch} , ", end='')
                print(f"Loss: {losses.avg} , Accuracy: {accuracies.avg}")
                # Print out f1 score per class for more detail if specified
                if log_f1:
                    report_f1_batch(pred_batch, label_batch, class_names)
        except Exception as e:
            print(f"Training batch failed due to Exception: {e}")
            continue
    
    # Calculate the epoch f1 score. Note f1 score can't be averaged over the epoch, it has to be
    # calculated using the accumulated labels and predictions over an epoch
    pred_epoch = cat(pred_list).cpu().numpy().tolist()
    label_epoch = cat(label_list).cpu().numpy().tolist()
    f1_epoch = f1_score(pred_epoch, label_epoch, average='weighted')
    
    # Writing epoch results to tensorboard
    writer.add_scalar('Train/Epoch/Loss', losses.avg, epoch+iter_)
    writer.add_scalar('Train/Epoch/Accuracy', accuracies.avg, epoch+iter_)
    writer.add_scalar('Train/Epoch/f1', f1_epoch, epoch+iter_)
    
    report_epoch = classification_report(label_epoch, pred_epoch)
    print(report_epoch)
    
    # Record epoch results in log file if path specified
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'\nTraining on Epoch {epoch} \n')
            f.write(f'Average loss: {losses.avg} \n')
            f.write(f'Average accuracy: {accuracies.avg} \n')
            f.write(f'F1 score: {f1_epoch} \n\n')
            f.write(report_epoch)
            f.write('*' * 25)
            f.write('\n')

    return losses.avg, accuracies.avg, f1_epoch


#####################################################################
# Evaluation

def save_model(model: CharCNN, model_path: str,
               model_prefix: str, epoch: int,
               max_length: int, lr: float,
               loss: float, acc: float, 
               f1: float, type_: str):
    model_str = f"model_{model_prefix}"
    epoch_str = f"epoch_{epoch}"
    maxlen_str = f"maxlen_{max_length}"
    lr_str = f"lr_{lr}"
    loss_str = f"loss_{loss:.4f}"
    acc_str = f"acc_{acc:.4f}"
    f1_str = f"f1_{f1:.4f}"
    model_name = f"{'_'.join([model_str, epoch_str, maxlen_str, lr_str, loss_str, acc_str, f1_str])}.pth"
    t_save(model.state_dict(), os.path.join(model_path, f"{type_}_{model_name}"))

def _evaluate_epoch(model: CharCNN, test_dl: DeviceDataLoader,
                    criterion: CrossEntropyLoss,
                    epoch: int,
                    writer: SummaryWriter,
                    class_names: List[Union[str, int]]=[0,1,2],
                    log_file: str=None,
                    log_f1: bool=True,
                    print_every: int=100):
    
    # Initial evaluation setup
    model.eval()
    iter_per_epoch, progress_bar = _setup_progress(test_dl)
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pred_list = list()
    label_list = list()
    for iter_, batch in progress_bar:
        # For evaluation, disable gradient calculations
        # This should speed up computation
        # Also would avoid leaking test data into back prop computation graph
        try:
            with no_grad():
                pred_label, scores = process_batch(model, batch, None, 
                                                   criterion, train=False)
            for k in scores:
                if 'loss' in k:
                    loss_batch = scores[k]
                elif 'acc' in k:
                    acc_batch = scores[k]
            pred_batch, label_batch = pred_label
            pred_list.append(pred_batch)
            label_list.append(label_batch)

            # Writing batch results to tensorboard
            writer.add_scalar('Test/Batch/Loss', loss_batch, epoch*iter_per_epoch+iter_)
            writer.add_scalar('Test/Batch/Accuracy', acc_batch, epoch*iter_per_epoch+iter_)
            pred_np = pred_batch.cpu().numpy().tolist()
            label_np = label_batch.cpu().numpy().tolist()
            f1_batch = f1_score(pred_np, label_np, average='weighted')
            writer.add_scalar('Test/Batch/f1', f1_batch, epoch*iter_per_epoch+iter_)

            # Log losses and accuracy
            # sum losses and accuracies over all batches in 1 epoch and calculated the average
            losses.update(loss_batch) 
            accuracies.update(acc_batch)

            # Print out training results at fixed batch intervals specified by 'print_every'
            if (iter_ % print_every == 0) and (iter_ > 0):
                print(f"\n[Validation - Epoch: {epoch+1}] , ", end='')
                print(f"Iteration: {iter_}/{iter_per_epoch} , ", end='')
                print(f"Loss: {losses.avg} , Accuracy: {accuracies.avg}")
                # Print out f1 score per class for more detail if specified
                if log_f1:
                    report_f1_batch(pred_batch, label_batch, class_names)
        except Exception as e:
            print(f"Testing batch failed due to Exception: {e}")
            continue
    
    # Calculate the epoch f1 score. Note f1 score can't be averaged over the epoch, it has to be
    # calculated using the accumulated labels and predictions over an epoch
    pred_epoch = cat(pred_list).cpu().numpy().tolist()
    label_epoch = cat(label_list).cpu().numpy().tolist()
    f1_epoch = f1_score(pred_epoch, label_epoch, average='weighted')
    
    # Writing epoch results to tensorboard
    writer.add_scalar('Test/Epoch/Loss', losses.avg, epoch+iter_)
    writer.add_scalar('Test/Epoch/Accuracy', accuracies.avg, epoch+iter_)
    writer.add_scalar('Test/Epoch/f1', f1_epoch, epoch+iter_)
    
    report_epoch = classification_report(label_epoch, pred_epoch)
    print(report_epoch)
    
    # Record epoch results in log file if path specified
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'\nValidation on Epoch {epoch} \n')
            f.write(f'Average loss: {losses.avg} \n')
            f.write(f'Average accuracy: {accuracies.avg} \n')
            f.write(f'F1 score: {f1_epoch} \n\n')
            f.write(report_epoch)
            f.write('*' * 25)
            f.write('\n')

    return losses.avg, accuracies.avg, f1_epoch

    
#####################################################################
# Pipeline

def pipeline_process(model: CharCNN, 
                     train_dl: DeviceDataLoader,
                     test_dl: DeviceDataLoader,
                     labels: Series, 
                     max_length: int,
                     writer: SummaryWriter,
                     optimiser: str='sgd',
                     unbalance_classes: bool=False, 
                     lr: float=0.01, momentum: float=0.9, 
                     schedule_lr: bool=True,
                     num_epoch: int=50, 
                     log_file: str=None,
                     log_f1: bool=True,
                     print_every: int=100,
                     checkpoint: bool=True, 
                     model_folder: str=None,
                     model_prefix: str=None,
                     early_stop: bool=True,
                     patience: int=3):
    """
    :param unbalance_classes: Indicator for initialising a weighted crossentropy loss function.
    :type unbalance_classes: bool
    """
    # Initialise optimisation algorithm
    class_weights = get_class_weights(labels) if unbalance_classes else None
    class_names = list(dict(Counter(labels)).keys())
    optimiser, criterion, scheduler = _init_optimisation(model, optimiser, class_weights, 
                                                         lr, momentum, schedule_lr)
    
    # Training and evaluation
    best_f1 = 0
    best_epoch = 0
    for epoch in range(num_epoch):
        # Training
        train_loss, train_acc, train_f1 = _train_epoch(model, train_dl, optimiser, 
                                                       criterion, epoch, writer, class_names, 
                                                       log_file, log_f1, print_every)
        
        # Validation
        test_loss, test_acc, test_f1 = _evaluate_epoch(model, test_dl, criterion, epoch, 
                                                       writer, class_names, log_file, log_f1, 
                                                       print_every)
        
        # Learning rate update
        if schedule_lr:
            scheduler.step() # There was a major update post torch 1.1.0, 
                             # lr scheduler must be called after optimiser update now
        lr = optimiser.state_dict()["param_groups"][0]["lr"]
        
        # Report on epoch loss and accuracy for train and test
        print(f"[Epoch: {epoch+1} / {num_epoch}]\t", end='')
        print(f"train_loss: {train_loss:.4f} \ttrain_acc: {train_acc:.4f} \t", end='')
        print(f"val_loss: {test_loss:.4f} \tval_acc: {test_acc:.4f}")
        print("=" * 50)
        
        # model checkpoint based on f1 scoring, performed every epoch
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch
            if checkpoint:
                save_model(model, model_folder, model_prefix, epoch, 
                           max_length, lr, test_loss, test_acc, test_f1, "CHECK")
        
        # Early stopping if specified
        if early_stop:
            if epoch - best_epoch > patience > 0:
                print(f"Stop training at epoch {epoch+1}. The lowest loss achieved is {test_loss:.4f} at epoch {best_epoch}")
                break
    
    save_model(model, model_folder, model_prefix, epoch, 
               max_length, lr, test_loss, test_acc, test_f1, "FINAL")

#####################################################################
# Main
                      
if __name__ == "__main__":
    pass