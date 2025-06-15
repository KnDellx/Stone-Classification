from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from dataset import StoneDataset
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor,Compose,RandomHorizontalFlip, Normalize, ToPILImage, RandomRotation, ColorJitter, FiveCrop, Lambda
from torchvision.models import ViT_B_16_Weights
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings("ignore")


def predict(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []  # Store predicted classes
    image_ids = []    # Store image filenames

    with torch.no_grad():  # Disable gradient computation
        for images, img_paths in tqdm(loader, desc="Predicting on test set"):
            images = images.to(device)  # Move images to the specified device
            outputs = model(images)     # Forward pass to get model outputs
            _, predicted = torch.max(outputs, 1)  # Get predicted classes
       
            # Collect predictions and image IDs
            predictions.extend(predicted.cpu().numpy())
            image_ids.extend([os.path.basename(path) for path in img_paths])

    return image_ids, predictions
def predict_with_tta(model, test_loader, device, tta_transforms_list):
    model.eval()
    predictions = []  # 存储预测结果
    image_ids = []    # 存储图像文件名
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='TTA Predicting...')
        for pil_images_list, img_paths in progress_bar:
            batch_size = len(pil_images_list)
            # 初始化累计预测概率张量
            batch_outputs_sum = torch.zeros(batch_size, model.heads.head.out_features \
                                             if not isinstance(model.heads.head, nn.Sequential) else model.heads.head[-1].out_features).to(device)
            for tta_transform in tta_transforms_list:
                is_multi_crop = any(isinstance(t, (FiveCrop, transforms.TenCrop)) 
                                    for t in (tta_transform.transforms if isinstance(tta_transform, Compose) else [tta_transform]))
                if is_multi_crop:
                    all_crops = []
                    for img in pil_images_list:
                        crops_tensor = tta_transform(img)  # (n_crops, C, H, W)
                        all_crops.append(crops_tensor)
                    augmented = torch.stack(all_crops).to(device)  # (B, n_crops, C, H, W)
                    bs, n_crops, C, H, W = augmented.shape
                    augmented_for_model = augmented.view(bs * n_crops, C, H, W)
                else:
                    augmented_for_model = torch.stack([tta_transform(img) for img in pil_images_list]).to(device)
                    n_crops = 1
                outputs = model(augmented_for_model)  # (B*n_crops, num_classes)
                if is_multi_crop:
                    outputs = outputs.view(batch_size, n_crops, -1).mean(dim=1)  # 平均各个作物的结果
                # 累计 softmax 后预测概率
                batch_outputs_sum += F.softmax(outputs, dim=1)
            avg_outputs = batch_outputs_sum / len(tta_transforms_list)
            _, preds = torch.max(avg_outputs, 1)
            predictions.extend(preds.cpu().numpy())
            image_ids.extend([os.path.basename(path) for path in img_paths])
    return image_ids, predictions


def pil_list_collate(batch):
    # batch 是一个列表，每个元素是 (PIL.Image, label)
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]  # 保持标签为字符串或原始类型
    return images, labels

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


if __name__ == "__main__":

    from torchvision import transforms
    print("Loading dataset......")
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

    # 加载数据集
    dataset_test = StoneDataset(root="./dataset/test", split="test", transforms=None)   # for Kaggle test only
    print(f"Test size: {len(dataset_test)}")
    test_loader = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False, num_workers = 8, pin_memory = True, collate_fn=pil_list_collate)
    print("Loading model......")
    model = torchvision.models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 3)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model.load_state_dict(torch.load("configs/model_epoch_noCenterCrop.pth")['model_state_dict'])
    model = model.to(device)
    
    print("Testing model......")
    image_ids, predictions = predict_with_tta(model, test_loader, device, tta_transform_list)

    # Create DataFrame
    submission_df = pd.DataFrame({
       "id": image_ids,    # Image filenames
       "label": predictions  # Predicted classes
    })

    # Save to the specified path
    OUTPUT_DIR = "logs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Kaggle submission file saved to {submission_path}")
    




