import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import json

def train_one(model, dataloader, criterion, logger, optimiser, epoch, device, cuda=True):
    results = {}
    loss = 0.0
    num = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimiser.zero_grad()
        outs = model(data)
        losses = torch.mean(criterion(outs, target))
        losses.backward()
        optimiser.step()
        loss += losses
        num += 1
        if not batch_idx % 100:
            print("running loss: {}".format(loss / num))
    return loss


class L1criterion(nn.Module):
    def __init__(self):
        super(L1criterion, self).__init__()
    def forward(self, outputs, targets):
        return torch.abs(targets - outputs).mean()

def train(model, dataloader, epoch_scheduler, logger, epochs, checkpoint, histfile, cuda=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = []
    criterion = L1criterion()
    for epoch in range(epochs):
        optimiser = optim.SGD(model.parameters(), lr=epoch_scheduler(epoch), momentum=0.9)
        loss = train_one(model, dataloader, criterion, logger, optimiser, epoch, device, cuda=cuda)
        history.append({'epoch': epoch, 'loss': str(loss.data.cpu().numpy()), 'lr': epoch_scheduler(epoch)})
        print("hist: {}".format(history[-1]))
        torch.save(model, checkpoint)
        with open(histfile, 'w') as h:
            h.write(json.dumps(history))
    return history
