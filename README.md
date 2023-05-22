# Applied-Machine-Learning
This readme document can help you run quickly.
args.py is the class that sets the parameters. All parameter Settings for this report can be found here.
train.py is used to train the model. This report will provide a running example of colab below, which is also the best hyperparameter example tested in this report.
test.py is used to test the model, and this report also provides an example below.
dataset folder is where the datasets are stored, where Place2_simp can be stored, and where a dataset loader has been provided.
logs are the directory where all log files are stored.
loss is the place where all loss functions will be stored.
models are the place where all mdel will be stored.
optimizer is the place where all optimeizer will be stored.
utils is place where all tool using the report will be stored.

An example of colab is provided below:
!python train.py \
--model resnet34 \
--dataset-root dataset/Places2_simp/ \
--width 224 \
--height 224 \
--epochs 30 \
--train-mode train \
--gpu True \
--num-workers 126 \
--loss-name CrossEntropyLoss \
--batch-size 256 \
--train-size-len 0.8 \
--optimizer SGD \
--lr 0.001 \
--last-lr 0.001 \
--weight-decay 0.001 \
--momentum 0.9 \
--num-samples 20 \
--save-path /content/drive/MyDrive/Machine-learning/logs/lr001-llr001-SGD-epochs30-bs256-m9-wd001 \
--result-name trained-model 

!python test.py \
--model resnet34 \
--dataset-root dataset/testset/ \
--train-mode test \
--width 224 \
--height 224 \
--gpu True \
--num-workers 4 \
--num-samples 20 \
--batch-size 256 \
--save-path /content/drive/MyDrive/Machine-learning/logs/lr001-llr001-SGD-epochs30-bs256-m9-wd001 \
--result-name trained-model 
