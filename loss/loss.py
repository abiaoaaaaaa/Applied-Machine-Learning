import torch.nn as nn

##########################
# Load loss function
##########################

# Cross entropy loss
def CrossEntropyLoss():
    return nn.CrossEntropyLoss()

# Passing in the name of the loss function returns the loss function
# loss_name: the name of the loss function
def make_loss(loss_name):
    if loss_name == "CrossEntropyLoss":
        return CrossEntropyLoss()
    else:
        raise Exception("Unknown loss name: {}".format(loss_name))
