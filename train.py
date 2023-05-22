from utils.logger import Log
from models.resnet import make_model
from optimizer.optimizers import make_optimizers
from loss.loss import make_loss
from args import argument_parser, make_opti_args
from dataset.places2 import places_train
from utils.avgmeter import AverageMeter
from utils.eval import accuracy, accuracy_top
from utils.utils import Args
from utils.mk_file import create_folder

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch 
import warnings
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import os
warnings.filterwarnings("ignore", category=UserWarning)

########################################
# train model
########################################

# get args
args = argument_parser()
# create log
if args.train_mode == "train":
  create_folder(args.save_path)
# create tensorboard writer
writer = SummaryWriter(log_dir=args.save_path)
# create device
device = "cpu"
# is use gpu, device = "cuda"
if args.gpu:
    if torch.cuda.is_available():
        device = "cuda"
# load class label name
class_names = os.listdir(args.dataset_root)

# train model
# model: need to train model
# train_loader: train data loader
# valid_loader: valid data loader
# log: log
def train(model, train_loader, valid_loader, log):
    # to get loss function
    criterion = make_loss(args.loss_name)
    # to get optimizer
    optimizer = make_optimizers(args.optimizer,model=model,args=Args(make_opti_args()))

    # train start
    for epoch in range(args.epochs):
        model.train()
        loss_ave_trian = AverageMeter()
        acc_ave_train  = AverageMeter()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_ave_trian.update(loss.item())
            acc_ave_train.update(accuracy(outputs, labels))
            # write loss and acc to log file
            log.log("epoch: {}[{}] --batch: {}[{}] --loss: {:.2f} --train_acc: {:.2f}%"
            .format(epoch + 1, 
            args.epochs,
            i + 1, 
            len(train_loader), 
            loss_ave_trian.val, 
            acc_ave_train.val))
        # write loss and acc to tensorboard
        writer.add_scalar('train/loss(avg)', loss_ave_trian.avg, epoch)
        writer.add_scalar('train/acc(avg)', acc_ave_train.avg, epoch)
        loss_ave_trian.reset()
        acc_ave_train.reset()

        # valid start
        model.eval()
        all_preds = []
        all_labels = []
        loss_ave_valid = AverageMeter()
        acc_ave_valid = AverageMeter()
        acc1_ave_valid = AverageMeter()
        acc5_ave_valid = AverageMeter()
        with torch.no_grad():
            for i, data in enumerate(valid_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc1,acc5 = accuracy_top(outputs,labels,topk=(1,5))
                acc1_ave_valid.update(acc1.item())
                acc5_ave_valid.update(acc5.item())
                _,preds = torch.max(outputs,1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                loss_ave_valid.update(loss.item())
                acc_ave_valid.update(accuracy(outputs, labels))
        # write loss and acc to log file
        log.log("epoch: {}[{}] --loss: {:.2f} --valid_acc(avg): {:.2f}% --valid_top1_acc(avg): {:.2f}% --valid_top5_acc(avg): {:.2f}%"
        .format(
          epoch + 1,
          args.epochs,
          loss_ave_valid.avg,
          acc_ave_valid.avg,
          acc1_ave_valid.avg,
          acc5_ave_valid.avg
        ))
        log.log("#########################################")
        # write loss and acc to tensorboard
        writer.add_scalar('valid/top1_acc(avg)', acc1_ave_valid.avg, epoch)
        writer.add_scalar('valid/top5_acc(avg)', acc5_ave_valid.avg, epoch)
        writer.add_scalar('valid/loss(avg)', loss_ave_valid.avg, epoch)
        writer.add_scalar('valid/acc(avg)', acc_ave_valid.avg, epoch)
        loss_ave_valid.reset()
        acc_ave_valid.reset()
        acc1_ave_valid.reset()
        acc5_ave_valid.reset()
        # plot confusion matrix
        cm = confusion_matrix(all_labels,all_preds)
        plt.figure(figsize=(20,20))
        sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = Image.open(buf)
        image = transforms.ToTensor()(image)
        writer.add_image("Confusion matrix(valid)/["+str(epoch)+"]",image)

    # random select image to write in tensorboard
    indices = np.random.choice(len(valid_loader.dataset), args.num_samples, replace=False)
    for i in indices:
      inputs, labels = valid_loader.dataset[i]
      inputs = inputs.to(device)
      inputs = inputs.unsqueeze(0)
      outputs = model(inputs)
      probabilities = F.softmax(outputs, dim=1).cpu().detach().numpy()
      top5_preds = np.argsort(probabilities)[0, -5:]
      actual_class_name = class_names[labels]
      predicted_class_names = [class_names[i] for i in top5_preds]
      mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
      std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
      inputs = inputs.view(3, 224, 224)
      inputs_unnorm = std * inputs.cpu().numpy() + mean
      inputs_unnorm = np.clip(inputs_unnorm, 0, 1)
      inputs_image = Image.fromarray((inputs_unnorm * 255).astype(np.uint8).transpose(1, 2, 0))
      plt.figure()
      plt.title(f"True: {actual_class_name}\nPred: {', '.join(predicted_class_names)}")
      plt.imshow(inputs_image)
      buf = io.BytesIO()
      plt.savefig(buf, format='jpeg')
      buf.seek(0)
      image = Image.open(buf)
      image = transforms.ToTensor()(image)
      writer.add_image("True label with top-5 predictions(valid)/["+str(i)+"]",image)
    return model

# pre-conditioning before training
# log: log file
def pre(log):
    log.log("#########################################")
    log.log("======Prepare train and valid dataset======")
    log.log("#########################################")
    transform = transforms.Compose([
    transforms.Resize((args.width, args.height)),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop((args.width, args.height), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader, valid_loader = places_train(root=args.dataset_root, 
                          transform=transform, 
                          train_size_len=args.train_size_len, 
                          batch_size=args.batch_size, 
                          shuffle=args.shuffle, 
                          num_workers=args.num_workers)

    log.log("#########################################")
    log.log("======train image: "+str(len(train_loader.dataset))+"======")
    log.log("======valid image: "+str(len(valid_loader.dataset))+"======")
    log.log("#########################################\n")

    log.log("#########################################")
    log.log("======Prepare model======")
    log.log("#########################################\n")
    model = make_model(args.model, num_class=args.num_classes)
    log.log(model)
    return train_loader, valid_loader, model
    
# main function
if __name__ == '__main__':
    log = Log(args.save_path, args.result_name)
    log.open_log_file()
    train_loader, valid_loader, model = pre(log)
    log.log(args)

    log.log("#########################################")
    log.log("======Starting train======")
    log.log("#########################################\n")
    model = model.to(device)
    model = train(model=model,train_loader=train_loader,valid_loader=valid_loader,log=log)

    log.log("#########################################")
    log.log("======Saving model======")
    log.log("#########################################\n")
    torch.save(model.state_dict(), args.save_path+"/"+args.result_name+".pth")
    
    log.log("#########################################")
    log.log("======Finished train======")
    log.log("#########################################\n")
    log.close_log_file()

    

