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
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--vis_interval', type=int, default=100)
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

dataset_train = CirclesLoad(args.root,  img_tf, 'train')
dataset_val = CirclesLoad(args.root,  img_tf, 'train')

print (len(dataset_train))
mini_batch = 512

iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=mini_batch,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=4))

model = vgg13().cuda()
device = torch.device('cuda')
import torch.nn as nn

criterion = nn.BCELoss().cuda()
num_labels = 2
optimizer = optim.Adam(model.parameters(), lr=.00001, betas=(0.5, 0.999))

def my_loss(logits, gtlbls):
    assert logits.shape[0] == mini_batch
    loss =  torch.mean(torch.abs(logits-gtlbls))
    return loss

    
for i in tqdm(range(0, 300)):
    image, gtlbls = [x.to(device) for x in next(iterator_train)]
    if image.shape[0] != mini_batch:
        print('breaking out of short img')
        continue
    model.train()
    optimizer.zero_grad()
    #y_ = (torch.rand(mini_batch, 1) * num_labels).type(torch.cuda.LongTensor).squeeze()
    
    output = model(image)
    loss = my_loss(output, gtlbls)
    if i%100 == 0:
        print("loss",loss)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "xyrmodel.pth")


#    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
#        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
#                  [('model', model)], [('optimizer', optimizer)], i + 1)