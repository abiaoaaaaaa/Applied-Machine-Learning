##########################
# Load optimizers
##########################
import torch.nn as nn
import torch.optim as optim
    
def SGD(model, 
        lr = 0.001, 
        last_lr = 0.001,
        momentum=0.9, 
        weight_decay=5e-4, 
        dampening = 0,
        nesterov = False):
  fc_ids = [id(p) for p in model.fc.parameters()]
  other_params = [p for p in model.parameters() if id(p) not in fc_ids]
  return optim.SGD([
    {"params":model.fc.parameters(),"lr":last_lr},
    {"params":other_params,"lr":lr}], momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov)

def Adam(model, 
         lr = 0.001,
         last_lr = 0.001,
         betas=(0.9, 0.999),
         eps=1e-08,
         weight_decay=0,
         amsgrad=False):
  fc_ids = [id(p) for p in model.fc.parameters()]
  other_params = [p for p in model.parameters() if id(p) not in fc_ids]
  return optim.Adam([
    {"params":model.fc.parameters(),"lr":last_lr},
    {"params":other_params,"lr":lr}], betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

def RMSprop(model,
            lr = 0.001,
            last_lr = 0.001,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False):
  fc_ids = [id(p) for p in model.fc.parameters()]
  other_params = [p for p in model.parameters() if id(p) not in fc_ids]
  return optim.RMSprop([
    {"params":model.fc.parameters(),"lr":last_lr},
    {"params":other_params,"lr":lr}], alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)

def Adagrad(model,
            lr = 0.001,
            last_lr = 0.001,
            lr_decay=0,
            weight_decay=0,
            initial_accumulator_value=0,
            eps=1e-10):
  fc_ids = [id(p) for p in model.fc.parameters()]
  other_params = [p for p in model.parameters() if id(p) not in fc_ids]
  return optim.Adagrad([
    {"params":model.fc.parameters(),"lr":last_lr},
    {"params":other_params,"lr":lr}], lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value, eps=eps)


# Passing in the name of the optimizer returns the optimizer
# optimizer_name: the name of the optimizer
# model: the model to be optimized
# args: the arguments of the optimizer(to watch args.py in detals)
def make_optimizers(optimizer_name, 
                    model, 
                    args):
    if optimizer_name == "SGD":
        return SGD(model, 
                   lr=args.lr, 
                   last_lr=args.last_lr,
                   momentum=args.momentum, 
                   weight_decay=args.weight_decay, 
                   dampening=args.dampening, 
                   nesterov=args.nesterov)
    
    elif optimizer_name == "Adam":
        return Adam(model, 
                    lr=args.lr, 
                    last_lr=args.last_lr,
                    betas=args.betas, 
                    eps=args.eps, 
                    weight_decay=args.weight_decay, 
                    amsgrad=args.amsgrad)
    
    elif optimizer_name == "RMSprop":
        return RMSprop(model, 
                       lr=args.lr,
                       last_lr=args.last_lr,
                       alpha=args.alpha,
                       eps=args.eps,
                       weight_decay=args.weight_decay,
                       momentum=args.momentum,
                       centered=args.centered)
    
    elif optimizer_name == "Adagrad":
        return Adagrad(model, 
                       lr=args.lr,
                       last_lr=args.last_lr,
                       lr_decay=args.lr_decay, 
                       weight_decay=args.weight_decay, 
                       initial_accumulator_value=args.initial_accumulator_value, 
                       eps=args.eps)
    else:
        raise Exception("Unknown optimizer name: {}".format(optimizer_name))
    
