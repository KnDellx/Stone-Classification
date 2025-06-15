'''
import package and configs
'''
import time
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor,Compose,RandomHorizontalFlip, Normalize, ToPILImage, RandomRotation, ColorJitter
from torchvision.models import ViT_B_16_Weights
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from dataset import StoneDataset
from timm import create_model
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings("ignore")
from train import CenterAwareOnly, SaliencyBGBlur


'''
use evaluation
'''
def evaluate_model(model, test_loader, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='test results is: ')
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        print(f'Accuracy on test data: {100 * correct / total:.2f}%')





if __name__ == '__main__':
    print("Setting up...")
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    seed = 42
    random.seed(seed)  # Python 内置随机数种子
    np.random.seed(seed)  # NumPy 随机种子
    torch.manual_seed(seed)  # PyTorch 随机种子
    torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU 也要设置
    torch.backends.cudnn.deterministic = True  # 让 CNN 结果可复现
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    '''
    Pre-process the data
    '''
    # Define Transformations for the dataset
    print("Processing data...")


    transform_test = transforms.Compose([
        CenterAwareOnly(
        centercrop_size = 1500,
        output_size = 1000,
        alpha = 0.45
    ),
        # SaliencyBGBlur(thr=0.3),     # 显著性背景模糊
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.46107383, 0.45589945, 0.45033285),
                         (0.28263338, 0.28181302, 0.28723977))
        ])
    
    # Load CIFAR-10 dataset

    dataset_val = StoneDataset(
        root="./dataset/train_val", split="val", transforms=transform_test
    )
    
    test_loader = DataLoader(dataset=dataset_val, batch_size=64, shuffle=False, num_workers = 8, pin_memory = True)
    
    
    
    '''
    create and define the hyperparameters
    '''
    print("Creating model and config...")
    # Define Vision Transformer Model
    model = torchvision.models.vit_b_16(weights=None)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # start training initial schedule
    num_features = model.heads.head.in_features
    num_classes = 3 # 假设您的 StoneDataset 有3个类别，请根据实际情况修改
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.5), # 假设训练时有 Dropout，p 值应与训练时一致
        nn.Linear(num_features, num_classes)
    )    
    # model.heads.head = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(f"configs/model_epoch_CenterCrop_9.pth")['model_state_dict'])
    model.to(device)


    '''
    evaluate
    '''
    print('---------------------------------------------')

    
    # Evaluate the Model
    evaluate_model(model, test_loader, device)
    

