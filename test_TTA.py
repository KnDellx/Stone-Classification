'''
import package and configs
'''
import time
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor,Compose,RandomHorizontalFlip, Normalize, ToPILImage, RandomRotation, ColorJitter, CenterCrop, FiveCrop, Lambda
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
from train import CenterAwareOnly


'''
use evaluation
'''
def pil_list_collate(batch):
    # batch 是一个列表，每个元素是 (PIL.Image, label)
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels # 返回 PIL Image 列表和标签张量

# 修改 evaluate_model 以支持 TTA
def evaluate_model_with_tta(model, test_loader, device, tta_transforms_list):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='TTA test results is: ')
        for pil_images_list, labels in progress_bar: # images 是一个 batch 的原始图像
            labels = labels.to(device)
            batch_size = len(pil_images_list) # batch_size 是列表的长度
            
            # 存储每个图像的 TTA 预测概率
            batch_outputs_sum = torch.zeros(batch_size, model.heads.head.out_features if not isinstance(model.heads.head, nn.Sequential) else model.heads.head[-1].out_features).to(device) # 假设 num_classes 是 model.heads.head 的输出维度
            
            for tta_transform in tta_transforms_list:
                # 处理 FiveCrop/TenCrop 的特殊情况，它们返回多个作物
                current_tta_batch_tensors = []
                is_multi_crop_transform = any(isinstance(t, (FiveCrop, transforms.TenCrop)) for t in (tta_transform.transforms if isinstance(tta_transform, Compose) else [tta_transform]))

                if is_multi_crop_transform:
                    # 对于 FiveCrop/TenCrop，tta_transform(img) 会返回一个 tensor (n_crops, C, H, W)
                    # 我们需要分别处理每个图像的多个作物
                    # 为了简化，这里假设 tta_transform 已经包含了 Lambda(lambda crops: torch.stack(...))
                    # 使得 tta_transform(img) 返回一个 (N_CROPS, C, H, W) 的张量
                    
                    # 收集一个批次中所有图像的所有作物的张量
                    all_crops_for_batch = []
                    for img_pil in pil_images_list:
                        crops_tensor = tta_transform(img_pil) # 期望返回 (N_CROPS, C, H, W)
                        all_crops_for_batch.append(crops_tensor)
                    
                    # augmented_images 的形状将是 (B, N_CROPS, C, H, W)
                    augmented_images = torch.stack(all_crops_for_batch).to(device)
                    
                    bs_tta, n_crops_tta, c_tta, h_tta, w_tta = augmented_images.shape
                    augmented_images_for_model = augmented_images.view(bs_tta * n_crops_tta, c_tta, h_tta, w_tta)
                else:
                    # 对于非 multi-crop 变换，tta_transform(img) 返回 (C, H, W)
                    augmented_images_for_model = torch.stack([tta_transform(img_pil) for img_pil in pil_images_list]).to(device)
                    n_crops_tta = 1 # 非 multi-crop 视为只有1个"crop"

                outputs = model(augmented_images_for_model) # (B*N_CROPS, num_classes)
                
                if is_multi_crop_transform: # 或者 n_crops_tta > 1
                    outputs = outputs.view(batch_size, n_crops_tta, -1) # (B, N_CROPS, num_classes)
                    outputs = outputs.mean(dim=1) # (B, num_classes) - 平均 N_CROPS 的预测
                # else: outputs 已经是 (B, num_classes)

                batch_outputs_sum += F.softmax(outputs, dim=1)

            # 平均所有 TTA 变换的概率
            avg_outputs = batch_outputs_sum / len(tta_transforms_list)
            _, predicted = torch.max(avg_outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on test data with TTA: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    print("Setting up...")
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # 在输入大小不变时可以设为True以加速
    torch.cuda.empty_cache()
    
    print("Processing data...")

    # --- 定义 TTA 变换 ---
    image_size = 224
    normalize = transforms.Normalize((0.46107383, 0.45589945, 0.45033285), (0.28263338, 0.28181302, 0.28723977))

    # 变换列表
    tta_transform_list = [
        # 1. 标准变换 (原始图像)
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ]),
        # 2. 水平翻转
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0), # 总是翻转
            transforms.ToTensor(),
            normalize
        ]),
        # 3. 中心裁剪 (如果原始图像比 image_size 大很多，可以考虑)
        transforms.Compose([
            transforms.Resize(int(image_size * 1.15)), # 先放大一点
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ]),
        # 4. FiveCrop (四个角点 + 中心点裁剪，然后平均) - 这个会显著增加计算量
        transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))), # 确保裁剪前图像略大于目标尺寸
            transforms.FiveCrop(image_size), # 返回一个包含5个 PIL Image 的元组
            Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(), normalize])(crop) for crop in crops])) # 对每个crop应用ToTensor和Normalize
        ]),
        # 5. 增加一点 ColorJitter (轻微的颜色抖动)
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            normalize
        ])
    ]

    # --- 数据加载器使用标准的测试变换 (不包含TTA列表中的变换) ---
    # TTA 是在 evaluate_model_with_tta 内部对每个原始图像应用的
    standard_test_transform = transforms.Compose([
        # 注意：这里 Resize 到一个稍大的尺寸，如果 TTA 中包含裁剪，可以提供更多信息
        # 但如果 TTA 主要是翻转和颜色变化，直接 Resize 到 image_size 即可
        transforms.Resize((image_size, image_size)), # 或者 Resize(256) 如果 TTA 中有 CenterCrop(224)
        # ToTensor 应该在 TTA 内部的每个变换中单独应用，因为 TTA 操作的是 PIL Image
        # transforms.ToTensor(), # 不在这里 ToTensor
        # normalize 也在 TTA 内部应用
        # normalize
    ])
    
    dataset_val = StoneDataset(
        root="./dataset/train_val", split="val", 
        transforms=None 
    )


    test_loader = DataLoader(dataset=dataset_val, batch_size=64, shuffle=False, num_workers = 8, pin_memory = True, collate_fn=pil_list_collate, prefetch_factor=2) # 减小 batch_size 以防 TTA 显存不足
    
    
    print("Creating model and config...")
    model = torchvision.models.vit_b_16(weights=None)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    num_features = model.heads.head.in_features
    num_classes = 3 
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_features, num_classes)
    )    
    # model.heads.head = nn.Linear(num_features, num_classes) # 这一行会覆盖上面的 Sequential，请检查是否需要


    # 加载模型权重
    checkpoint_path = "configs/model_epoch_20.pth" # 您的模型路径
    print(f"Loading model checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.to(device)

    print('---------------------------------------------')
    
    # Evaluate the Model with TTA
    evaluate_model_with_tta(model, test_loader, device, tta_transform_list)
