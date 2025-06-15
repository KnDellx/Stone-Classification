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
import logging
import os
from PIL import Image
import cv2
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 调试时启用同步模式

'''
create logging directory and realize logging function
'''
warnings.filterwarnings("ignore")
log_dir = 'log'
os.makedirs(log_dir, exist_ok = True)
log_file = os.path.join(log_dir, 'training_saliency.log')
logging.basicConfig(
    filename = log_file,
    filemode = 'a',
    format = '%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO
)

class CenterAwareOnly(torch.nn.Module):
    """
    Applies a center-aware overlay by blending a globally resized view
    with a zoomed-in center crop view of the image.
    """
    def __init__(self, centercrop_size, output_size, alpha = 0.5):
        super().__init__()
        self.centercrop_size = centercrop_size
        self.output_size = (output_size, output_size)
        self.alpha = alpha
        self.resizer = transforms.Resize(self.output_size)
        self.centercrop = transforms.CenterCrop(self.centercrop_size)
    def forward(self, img:Image.Image):
        # 1. Global View Processing
        global_view_img = self.resizer(img.copy()) # Use .copy() if img might be used later

        # 2. Center View Processing
        center_cropped_img = self.resizer(self.centercrop(img.copy()))
        global_view_img = global_view_img.convert('RGB')
        center_view_img = center_cropped_img.convert('RGB')
        blended_img = Image.blend(global_view_img, center_view_img, self.alpha)

        return blended_img
    def __repr__(self):
        return f'{self.__class__.__name__}(centercrop size is {self.centercrop_size}, output size is {self.output_size}, alpha is {self.alpha})'
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

def train_model(model, train_loader, criterion, optimizer, device, epochs=10, start_epoch = 1, scheduler = None):
    start_time = time.time()
    model.train()
    scaler = GradScaler()  # 初始化混合精度缩放器

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        correct, total = 0, 0
        # set up progress bar to help me debug
        
        progress_bar = tqdm(train_loader, desc=f'Epoch{epoch + 1}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            with autocast():  # 启用混合精度
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # add learning rate scheduler to help converge
        if scheduler:
            scheduler.step()
        Epoch_time = time.time() - start_time 
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total:.2f}%")
        print(f'This epoch uses {Epoch_time:.2f} seconds')
        # logging info
        logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total:.2f}%")
        logging.info(f"This epoch uses {Epoch_time:.2f} seconds")
        save_checkpoint = {
        'epochs': epoch, # 保存的是已完成的 epoch 数 (0-indexed)
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }
        # if scheduler:
        #     save_checkpoint['scheduler_state_dict'] = scheduler.state_dict() # 保存 scheduler 状态
        
        torch.save(save_checkpoint, f"configs/model_epoch_CenterCrop_{epoch+1}.pth") # epoch+1 是为了文件名可读性
        torch.cuda.empty_cache()




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
    transform_train = transforms.Compose([
    CenterAwareOnly(
        centercrop_size = 500,
        output_size = 1000,
        alpha = 0.5
    ),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.2),  # 50% 概率水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomRotation(15),           # 旋转 ±15 度
    transforms.ToTensor(),
    transforms.Normalize((0.46020824, 0.4554496,  0.45052096), (0.28402117, 0.28318824, 0.28876383))
    ]) 

    transform_test = transforms.Compose([
        CenterAwareOnly(
        centercrop_size = 500,
        output_size = 1000,
        alpha = 0.5
    ),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.46020824, 0.4554496,  0.45052096), (0.28402117, 0.28318824, 0.28876383))
        ])
    
    # Load CIFAR-10 dataset
    dataset_train = StoneDataset(
        root="./dataset/train_val", split="train", transforms=transform_train
    )
    dataset_val = StoneDataset(
        root="./dataset/train_val", split="val", transforms=transform_test
    )
    
    train_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True, num_workers = 16, pin_memory = True,  prefetch_factor=2)
    test_loader = DataLoader(dataset=dataset_val, batch_size=64, shuffle=False, num_workers = 8, pin_memory = True)
    
    
    
    '''
    create and define the hyperparameters
    '''
    print("Creating model and config...")
    # Define Vision Transformer Model
    model = torchvision.models.vit_b_16(weights=None)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model.load_state_dict(torch.load("configs/vit_b_16-c867db91.pth"))
    num_classes = 3 # 假设您的 StoneDataset 有3个类别，请根据实际情况修改
    num_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p = 0.6),
        nn.Linear(num_features, num_classes)
    )
    model.to(device)
    
    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) 
    total_epochs_for_annealing = 10 # 与您 train_model 中的 epochs 参数一致
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_epochs_for_annealing, eta_min=1e-6) # eta_min 是可选的，默认为0
    '''
    train and evaluate
    '''
    try:
        # set the resume function
        print("Loading model...")
        resume_path = 'configs/model_epoch_90.pth'
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location = device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state loaded.")
            resume_epoch = checkpoint['epochs'] + 1
        else:
            pass
        print("Training...")
        train_model(model, train_loader, criterion, optimizer, device, epochs=10, start_epoch = 0, scheduler=scheduler)
    except RuntimeError as e:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        exit()
    print('---------------------------------------------')

    
    # Evaluate the Model
    evaluate_model(model, test_loader, device)
    

