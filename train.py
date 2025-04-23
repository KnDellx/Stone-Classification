'''
import package and configs
'''
import time
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor,Compose,RandomHorizontalFlip, Normalize, ToPILImage, RandomRotation, ColorJitter
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

'''
training progress
'''

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        # set up progress bar to help me debug
        
        progress_bar = tqdm(train_loader, desc=f'Epoch{epoch + 1}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        Epoch_time = time.time() - start_time 
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total:.2f}%")
        print(f'This epoch uses {Epoch_time:.2f} seconds')
        torch.save(model.state_dict(), f"configs/model_epoch_{epoch+1}.pth")
        torch.cuda.empty_cache()




if __name__ == '__main__':
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
    transform_train = transforms.Compose(
    [transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.2),  # 50% 概率水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomRotation(15),           # 旋转 ±15 度
    transforms.ToTensor(),
    transforms.Normalize((0.46020824, 0.4554496,  0.45052096), (0.28402117, 0.28318824, 0.28876383))]) 

    transform_test = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.46107383, 0.45589945, 0.45033285), (0.28263338, 0.28181302, 0.28723977))])
    
    # Load CIFAR-10 dataset
    dataset_train = StoneDataset(
        root="./dataset/train_val", split="train", transforms=transform_train
    )
    dataset_val = StoneDataset(
        root="./dataset/train_val", split="val", transforms=transform_test
    )
    
    train_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True, num_workers = 8, pin_memory = True)
    test_loader = DataLoader(dataset=dataset_val, batch_size=64, shuffle=False, num_workers = 8, pin_memory = True)
    
    
    
    '''
    create and define the hyperparameters
    '''
    # Define Vision Transformer Model
    model = create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=3
    )  # Using a pretrained ViT model
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = model.to(device)
    
    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)


    '''
    train and evaluate
    '''
    # try:
    #     train_model(model, train_loader, criterion, optimizer, device, epochs=20)
    # except RuntimeError as e:
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()
    #     exit()
    print('---------------------------------------------')
    epochs = 9
    model.load_state_dict(torch.load(f"configs/model_epoch_{epochs}.pth"))
    model.to(device)
    
    # Evaluate the Model
    evaluate_model(model, test_loader, device)


