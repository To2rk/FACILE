import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os, sys
import cv2
import random


labels_MalImg = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.gen!g', 
          'C2LOP.P', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 
          'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 
          'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']

labels_BIG2015 = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda', 'Tracur', 'Kelihos_ver1', 'Obfuscator.ACY', 'Gatak']

labels_Virus_MNIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def chooseDataset(dataset):
    if dataset == 'MalImg':
        labels = labels_MalImg
    elif dataset == 'BIG2015':
        labels = labels_BIG2015
    elif dataset == 'Virus_MNIST':
        labels = labels_Virus_MNIST
    else:
        print('Dataset load error, please check!')
        sys.exit()

    return labels

class MalwareImageDataset(Dataset):
    def __init__(self, dataset_path, dataset):

        self.dataset_path = dataset_path
        self.dataset = dataset
        self.images = os.listdir(self.dataset_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),    
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_index = self.images[index]
        img_path = os.path.join(self.dataset_path, image_index)

        image = self.transform(cv2.imread(img_path, cv2.IMREAD_COLOR))
        labels = chooseDataset(self.dataset)
        label = labels.index(image_index.split('-')[0])

        return image, label

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONSEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True 
