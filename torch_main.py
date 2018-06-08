import json
import argparse
import os
from torch.utils.data import DataLoader

from imgen_torch import ImageSequence
from torch_train import train
from models.torch_tunnel import DeepModel2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--minrate', type=float, default=0.0001)
    parser.add_argument('--maxrate', type=float, default=0.05)
    parser.add_argument('--lrdecay', type=float, default=0.8)
    parser.add_argument('--lrreset', type=int, default=7)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--history', type=str, default='history.json')

    args = parser.parse_args()
    model = DeepModel2()
    data_iter = ImageSequence(8, (20.0, 20.0), (7, 7))
    dataloader = DataLoader(data_iter, batch_size=8, workers=args.workers, shuffle=False)
    def scheduler(epoch):
        return (args.minrate + args.maxrate * (args.lrdecay**(epoch % args.lrreset)))
    history = train(model, scheduler, None, args.total_epochs, args.checkpoint)
    with open(args.history) as h:
        h.write(json.dumps(history))
