# 石头分类项目

本项目包含用于训练和评估石头分类模型的脚本。

## 文件说明与使用
从网盘中下载的模型文件请保存到configs文件夹底下，在以下陈述中我把网盘中的model_epoch_11.pth命名为model_epoch_noCenterCrop以避免混淆

### 1. 训练脚本 (`train.py`, `train_CenterAware.py`)

这些脚本用于训练石头分类模型。
**普通训练步骤：**
1. 把`model.load_state_dict(torch.load(f"configs/.pth")['model_state_dict'])`中的`.pth`修改为`vit_b_16-c867db91`,训练时请用这个文件
2. `model_epoch_i`是使用了saliency map的模型文件
`model_epoch_CenterCrop`是使用了`CenterAwareOnly`进行训练的模型文件，对应第二个大实验的Second Method
3. 训练前你需要确保
```
  model.heads.head = nn.Sequential(
        nn.Dropout(p = 0.6),
        nn.Linear(num_features, num_classes)
    )
```
  没有被注释掉
  
4. For certain kind of transform, you need to navigate to the `transfrom_train` part and uncomment the corresponding transform while comment another kind of transform.

**断点续训步骤：**

训练过程中，模型权重和优化器状态会定期保存到 `configs/` 目录下（例如 `model_epoch_X.pth`）。

要从某个断点继续训练：
1.  打开相应的训练脚本 (例如 `train.py`)。
2.  找到以下或类似的代码块：
    ```python
    # train.py
    resume_path = 'configs/model_epoch_11.pth'
    # train_CenterAware.py
    # resume_path = 'configs/model_epoch_90.pth'
    ```
3.  将 `resume_path` 的值修改为您希望从中恢复的 checkpoint 文件路径。
4. `start_epoch`参数赋值为resume_epoch
4.  脚本会自动加载模型、优化器（以及学习率调度器，如果已保存）的状态，并从下一个 epoch 开始继续训练。`start_epoch` 参数会根据 checkpoint 中的 `epochs` 自动设置。

### 2. 测试脚本 (`test.py`)

此脚本用于在测试集上评估已训练模型的性能。

**修改模型路径：**

要评估特定的模型：
1.  打开 `test.py` 文件。
2.  找到加载模型权重的行：
    ```python
    model.load_state_dict(torch.load(f"configs/model_epoch_29.pth")['model_state_dict'])
    ```
3.  将 `f"configs/model_epoch_29.pth"` 修改为您要测试的 `.pth` 模型文件的实际路径。
4. 4. For certain kind of transform, you need to navigate to the `transfrom_test` part and uncomment the corresponding transform while comment another kind of transform.

接下来如果你要测试`model_epoch_noCenterCrop.pth`，请：
1. 先解除注释
```
model.heads.head = nn.Linear(num_features, num_classes)
```
2. 运行脚本python test.py

如果你要测试除了`model_epoch_noCenterCrop.pth`以外的其他类型，请：
1. 先解除注释
```
 model.heads.head = nn.Sequential(
        nn.Dropout(p=0.5), # 假设训练时有 Dropout，p 值应与训练时一致
        nn.Linear(num_features, num_classes)
    )    
```
注释掉
```
model.heads.head = nn.Linear(num_features, num_classes)
```
2. 运行脚本python test.py


### 3. 带 TTA 的测试脚本 (`test_TTA.py`)

此脚本使用测试时增强 (Test-Time Augmentation, TTA) 来评估已训练模型的性能，通常能获得更鲁棒的结果。

**修改模型路径：**

要使用 TTA 评估特定的模型：
1.  打开 `test_TTA.py` 文件。
2.  找到定义模型权路径的行：
    ```python
    checkpoint_path = "configs/model_epoch_20.pth" # 您的模型路径
    ```
3.  将 `"configs/model_epoch_20.pth"` 修改为您要测试的 `.pth` 模型文件的实际路径。

如果你要测试`model_epoch_noCenterCrop.pth`，请：
1. 先解除注释
```
model.heads.head = nn.Linear(num_features, num_classes)
```
2. 将 `"configs/model_epoch_20.pth"` 修改为您要测试的 `.pth` 模型文件的实际路径。
3. 运行脚本python test_TTA.py

如果你要测试除了`model_epoch_noCenterCrop.pth`以外的其他类型，请：
1. 先解除注释
```
 model.heads.head = nn.Sequential(
        nn.Dropout(p=0.5), # 假设训练时有 Dropout，p 值应与训练时一致
        nn.Linear(num_features, num_classes)
    )    
```
2. 将 `"configs/.pth"` 修改为您要测试的 `.pth` 模型文件的实际路径。
3. 运行脚本python test_TTA.py

---

请确保您的数据集已按脚本内 `StoneDataset` 类预期的方式组织。
