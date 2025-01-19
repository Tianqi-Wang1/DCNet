import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import pickle as pkl
import math
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

#The maximum value of the absolute cosine value of the angle between any two basis vectors
threshold = 0.1

#Number of categories, which can also be interpreted as the number of basis vectors N that need to be generated, num_cls >= N
num_cls = 410

num_task = 10

num_per_class = int(num_cls // num_task)

#Dimension for basis vectors
dim = 256

#Slicing optimization is required due to insufficient memory
slice_size = 50   

#Optimize step numbers
step = 100000        
lr = 1e-4 #learning rate
save_name = './BV_pkl_CL/eq_100_4+10_256_0.1_'
if not os.path.exists('./BV_pkl_CL'):
    os.makedirs('./BV_pkl_CL')

def main():
    dtype = torch.float32
    for i in range(num_task):
        basis_vec = nn.Parameter(F.normalize(torch.randn((num_per_class, dim), dtype=dtype, device=device)))
        optim = torch.optim.SGD([basis_vec], lr=lr)
        if i != 0:
            old_vec = pkl.load(open(save_name+str(i-1)+'.pkl', 'rb')).cuda()


        pbar = tqdm(range(step), total=step)
        basis_vec.data.copy_(F.normalize(basis_vec.data))
        new_vec = copy.deepcopy(basis_vec.detach())
        if i == 0:
            target = basis_vec
        else:
            target = torch.cat((old_vec, new_vec))
        for _ in pbar:

            mx = threshold

            start = num_per_class * i
            end = num_per_class * (i + 1)
            e = F.one_hot(torch.arange(start, end, device=device), end)
            m = (basis_vec @ target.T).abs() - e
            mx = max(mx, m.max().item())
            loss = F.relu_(m - threshold).sum()

            loss.backward()
            basis_vec.data.copy_(F.normalize(basis_vec.data))
            new_vec = copy.deepcopy(basis_vec.detach())
            if i == 0:
                target = basis_vec
            else:
                target = torch.cat((old_vec, new_vec))

            if mx <= threshold + 0.0001:
                pkl.dump(target.data, open(save_name+str(i)+'.pkl', 'bw'))
                break
            optim.step()
            optim.zero_grad()
            pbar.set_description(f'{mx:.4f}')
    return


if __name__ == '__main__':
    if not os.path.exists(save_name):
        seed = 42
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        main()
