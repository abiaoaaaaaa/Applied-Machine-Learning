import torch

# accuracy of calculation
# outputs: output of model
# labels: labels of data
def accuracy(outputs, labels):
  with torch.no_grad():
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0) * 100

# accuracy of calculation
# outputs: output of model
# labels: labels of data
# topk: top k accuracy
def accuracy_top(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res