##########################
# Load resnet models
##########################

import torch
from torchvision import models

def resnet18(num_class = None):
    resnet18 = models.resnet18(pretrained=True)
    if num_class is not None:
        resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, num_class)
    return resnet18

def resnet34(num_class = None):
    resnet34 = models.resnet34(pretrained=True)
    if num_class is not None:
        resnet34.fc = torch.nn.Linear(resnet34.fc.in_features, num_class)
    return resnet34

def resnet50(num_class = None):
    resnet50 = models.resnet50(pretrained=True)
    if num_class is not None:
        resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_class)
    return resnet50

def resnet101(num_class = None):
    resnet101 = models.resnet101(pretrained=True)
    if num_class is not None:
        resnet101.fc = torch.nn.Linear(resnet101.fc.in_features, num_class)
    return resnet101

# Return the model through this function
# model_name: model name
# num_class: the number of class
def make_model(model_name, num_class = None):
    if model_name == "resnet34":
        return resnet34(num_class)
    elif model_name == "resnet18":
        return resnet18(num_class)
    elif model_name == "resnet50":
        return resnet50(num_class)
    elif model_name == "resnet101":
        return resnet101(num_class)
    else:
        raise Exception("Unknown model name: {}".format(model_name))
    