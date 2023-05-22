from utils.logger import Log
from models.resnet import make_model
from optimizer.optimizers import make_optimizers
from loss.loss import make_loss
from args import argument_parser, make_opti_args
from dataset.places2 import places_test
from utils.avgmeter import AverageMeter
from utils.eval import accuracy, accuracy_top
from utils.utils import Args

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch 
import warnings
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import torch.nn.functional as F
from PIL import Image
warnings.filterwarnings("ignore", category=UserWarning)

########################################
# test model
########################################

args = argument_parser()
device = "cpu"
if args.gpu:
    if torch.cuda.is_available():
        device = "cuda"
class_names = os.listdir(args.dataset_root)
writer = SummaryWriter(log_dir=args.save_path)

def pre(log):
    log.log("#########################################")
    log.log("======Prepare test dataset======")
    log.log("#########################################\n")
    transform = transforms.Compose([
    transforms.Resize((args.width, args.height)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_loader = places_test(args.dataset_root, transform, args.batch_size)
    log.log("======test image: "+str(len(test_loader.dataset))+"======\n")
    log.log("#########################################")
    log.log("======Prepare model======")
    log.log("#########################################\n")
    model = make_model(args.model, args.num_classes)
    return test_loader, model

if __name__ == "__main__":
    log = Log(args.save_path, args.result_name)
    log.open_log_file()
    test_loader, model = pre(log)
    model.load_state_dict(torch.load(args.save_path +"/"+ args.result_name+".pth"))
    model.eval()
    model = model.to(device)

    log.log("#########################################")
    log.log("======Starting test======")
    log.log("#########################################\n")
    all_preds = []
    all_labels = []
    acc1_ave_test = AverageMeter()
    acc5_ave_test = AverageMeter()
    acc_ave_test = AverageMeter()

    # test start
    with torch.no_grad():
      for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        acc_ave_test.update(accuracy(output,labels))
        acc1, acc5 = accuracy_top(output, labels, topk=(1,5))
        acc1_ave_test.update(acc1.item())
        acc5_ave_test.update(acc5.item())
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
      # log
      log.log("--test_top1_acc(avg):{:.2f}% --test_top5_acc(avg):{:.2f}% --test_acc(avg): {:.2f}%\n"
        .format(
          acc1_ave_test.avg,
          acc5_ave_test.avg,
          acc_ave_test.avg
        ))
      # to plot confusion matrix to tensorboard
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
      writer.add_image("Confusion matrix(Test)",image)
      # to plot top-5 predictions to tensorboard
      indices = np.random.choice(len(test_loader.dataset), args.num_samples, replace=False)
      for i in indices:
        inputs, labels = test_loader.dataset[i]
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
        writer.add_image("True label with top-5 predictions(Test)/["+str(i)+"]",image)
    log.log("#########################################")
    log.log("======Finish test======")
    log.log("#########################################\n")
    log.close_log_file()

