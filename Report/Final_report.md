# STA326 Project Report

## I Dataset

1. ### Data Source and Information

   Our data, consisting of 132213 labeled rock images, is acquired directly from geology researchers.  The dataset is made up of images of 3 main types of rocks: metamorphic rock, sedimentary rock, and igneous rock. The images are split into 3 subsets: training set with 112213 images, validation set and testing set with 15000 images respectively.

2. ### Exploratory Analysis

   - Category distributions:

     | Category                  | Metamorphic rock | Sedimentary rock | Igneous rock |
     | ------------------------- | ---------------- | ---------------- | ------------ |
     | Training set (unbalanced) | 24985            | 30896            | 46332        |
     | Validation set (balanced) | 5000             | 5000             | 5000         |

3. ### Data Preprocessing

   Images in both the training and validation sets are preprocessed.  We randomly flipped, rotated, and implemented color jitter on training images, and all of the images are resized to $$224 \times 224$$ and normalized with the mean/standard deviation values from each split:
   
   ```python
   transform_train = transforms.Compose(
       [transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(p=0.2),  
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
       transforms.RandomRotation(15),           
       transforms.ToTensor(),
       transforms.Normalize((0.46020824, 0.4554496,  0.45052096), 
                            (0.28402117, 0.28318824, 0.28876383))]) 
   
   transform_test = transforms.Compose(
       [transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize((0.46107383, 0.45589945, 0.45033285),
                            (0.28263338, 0.28181302, 0.28723977))])
   ```
   

## II Experiment I: use pre-trained weights

1. ### Model Selection

    We chose **ViT‑B/16** along with its pre-trained weights to leverage its global self‑attention for capturing long‑range dependencies in stone surface textures, aiming to improve the discrimination of subtle class differences that standard CNNs may miss. Its proven scalability on large image datasets made it a strong candidate for our ~102 K-image corpus.

2. ### Experimental Set-ups

   | Batch Size | Learning rate         | Learning rate scheduler | Optimizer |
   | ---------- | --------------------- | ----------------------- | --------- |
   | 64         | $$1 \times 10 ^{-5}$$ | StepLR, step = 5        | Adam      |

3. ### Result Analysis

   <img src=".\1.png" alt="1" style="zoom:33%;" />

   We "fine-tuned" the pre-trained model for 16 epochs and evaluated with the checkpoints saved after the last several epochs. Apparently, the model has over-fitted the training, and its performance on the validation set hardly improves during epoch 10-16.  Selecting checkpoints with the highest validation accuracies at epoch 11 (**75.22%**) and 13 (**75.13%**), we tested them and the testing set and obtained accuracies with **74.87%** and **75.00%,** respectively.
   
   Based on these results with a lot of improving spaces, we decided to seek more strategies for processing the training data, thus improving model performance.

## III Experiment II: operations on datasets

1. ### Central crop

   - #### Method

     xxxxxx

   - #### Results

     nnnnnn

2. ### Spectral residual map 

   - #### Method

     xxxxxx

   - #### Results

     nnnnnn

## IV Experiment III: Test-Time Adaptation (TTA)

1. ### Method

   xxxxxx

2. #### Results

   nnnnnn

## V Discussion & Summary



# Reference

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale*. In *International Conference on Learning Representations*. https://openreview.net/forum?id=YicbFdNTTy

Wightman, R. (2019). *PyTorch Image Models (timm)* [Computer software]. GitHub repository. https://github.com/rwightman/pytorch-image-models (doi: 10.5281/zenodo.4414861)