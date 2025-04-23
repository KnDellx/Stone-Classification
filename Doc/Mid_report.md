# STA326 Project Interim Report

## I Dataset

1. ### Data Source and Information

   Our data, consisting of 132213 labeled rock images, is acquired directly from geology researchers.  The dataset is made up of images of 3 main types of rocks: metamorphic rock, sedimentary rock, and igneous rock. The images are split into 3 subsets: training set with 112213 images, validation set and testing set with 15000 images respectively.

2. ### Exploratory Analysis

   - Category distributions:

     | Category       | Metamorphic rock | Sedimentary rock | Igneous rock |
     | -------------- | ---------------- | ---------------- | ------------ |
     | Training set   | 24985            | 30896            | 46332        |
     | Validation set | 5000             | 5000             | 5000         |

   - Means for each image channel:

     | Channel        | 1          | 2          | 3          |
     | -------------- | ---------- | ---------- | ---------- |
     | Training set   | 0.46020824 | 0.4554496  | 0.45052096 |
     | Validation set | 0.46107383 | 0.45589945 | 0.45033285 |

   - Standard deviation values for each image channel:

     | Channel        | 1          | 2          | 3          |
     | -------------- | ---------- | ---------- | ---------- |
     | Training set   | 0.28402117 | 0.28318824 | 0.28876383 |
     | Validation set | 0.28876383 | 0.28181302 | 0.28723977 |

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
   
   

## II Preliminary Results

1. ### Model Selection

    We chose **ViT‑B/16** to leverage its global self‑attention for capturing long‑range dependencies in stone surface textures, aiming to improve the discrimination of subtle class differences that standard CNNs may miss. Its proven scalability on large image datasets made it a strong candidate for our ~102 K-image corpus.

2. ### Experimental Set-ups

   | Batch Size | Learning rate         | Learning rate scheduler | Optimizer |
   | ---------- | --------------------- | ----------------------- | --------- |
   | 64         | $$1 \times 10 ^{-5}$$ |                         | Adam      |

   TODO: other special settings

3. ### Result Analysis

   TODO: 训练集表现，测试集表现图（loss/accuracy）及相关分析

## III Future Plans

To tackle the problem that the 

## IV Summary

xxxxxxx