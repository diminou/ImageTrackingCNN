import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

def train_one(model, dataloader, criterion, logger, optimiser, epoch, device, cuda=True):
    results = {}
    loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = [Variable(d) for d in data], [Variable(t) for t in target]
        if cuda:
            data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
        optimiser.zero_grad()
        outs = [model(d) for d in data]
        losses = sum([criterion(d, t) for (d, t) in zip(outs, target)]) / len(outs)
        losses.backward()
        optimiser.step()
        loss += losses
    return loss


def train(model, dataloader, epoch_scheduler, logger, epochs, checkpoint, cuda=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = []
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimiser = optim.SGD(model.parameters(), lr=epoch_scheduler(epoch), momentum=0.9)
        loss = train_one(model, dataloader, criterion, logger, optimiser, epoch, device, cuda=cuda)
        history.append({'epoch': epoch, 'loss': loss, 'lr': epoch_scheduler(epoch)})
        print("hist: {}".format(history[-1]))
        torch.save(model, checkpoint)
    return history
