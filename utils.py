import os, sys
sys.path.append(os.getcwd())

from PIL import Image
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import scipy.misc
from scipy.misc import imsave

class Dataset():
    def __init__(self,mode):
        path = 'F:\\torch_projects\\data\\Retinal\\train.txt' if mode=='train' else 'F:\\torch_projects\\data\\Retinal\\train.txt'
        self.data = []
        transform = []
        transform.append(T.CenterCrop(560))
        transform.append(T.Resize(512))
        transform.append(T.ToTensor())
        transform = T.Compose(transform)
        with open(path, 'r') as file_to_read:
            while(1):
                line = file_to_read.readline() # 整行读取数据
                if not line:
                    break
                image = Image.open('F:\\torch_projects\\data\\Retinal\\'+line.strip('\n'))
                self.data.append(transform(image))
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)
def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise, volatile=True)
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 28, 28)
    # print samples.size()

    samples = samples.cpu().data.numpy()