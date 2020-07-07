from tqdm import tqdm
import torch
import os
from torchvision import transforms
import argparse
from CirclesLoad import CirclesLoad
from vgg import *
from torch.utils import data
import torch.optim as optim
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch.nn as nn

class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0



parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/data/mialab/users/washbee/circles/')
args = parser.parse_args()
#if not os.path.exists(args.log_dir):
#    os.makedirs(args.log_dir)
size = (32, 32)
img_tf = transforms.Compose(
    [
        transforms.Resize(size=size),
        transforms.ToTensor()
    ]
)

dataset_val = CirclesLoad(args.root,  img_tf, 'val')

mini_batch = 512

iterator_test = iter(data.DataLoader(
    dataset_val, batch_size=mini_batch,
    sampler=InfiniteSampler(len(dataset_val)),
    num_workers=4))

device = torch.device('cuda')

criterion = nn.BCELoss().cuda()

model = vgg13()

checkpoint = torch.load("xyrmodel.pth")
model.load_state_dict(checkpoint)
model = model.to(device)


def my_loss(logits, gtlbls):
    assert logits.shape[0] == mini_batch
    loss =  torch.mean(torch.abs(logits-gtlbls))
    return loss

    
for i in tqdm(range(0, 1)):
    image, gtlbls = [x.to(device) for x in next(iterator_test)]
    if image.shape[0] != mini_batch:
        print('breaking out of short img')
        continue
    output = model(image)
    loss = my_loss(output, gtlbls)
    print("loss",loss.data.item())
    print("output",output)
    print("gt",gtlbls)

