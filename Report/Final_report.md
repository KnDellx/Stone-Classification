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
Before running further experiments, we did a few techinical improvements to  help prevent over-fitting, we present changes as follows:
* We add a dropout layer: `nn.Dropout(p = 0.6)` before linear layer
* we also use label smoothing: `nn.CrossEntropyLoss(label_smoothing=0.1)`
* In the second experiment, we think that model might stuck in local optimum, so we use a different scheduler `CosineAnnealingWarmRestarts`

We present reasons here:
* Dropout is used to prevent overfitting, ensuring the model does not rely too much on specific features, which is important for robust stone classification with limited or diverse data.
* Label smoothing helps the model generalize better by reducing overconfidence in predictions, which is useful when stone categories may have ambiguous boundaries.
* CosineAnnealingWarmRestarts allows the learning rate to periodically restart, helping the model escape local minima and explore better solutions in complex stone image datasets.
1. ### Blending Central Area 

   - #### First Method
      Our initial attempt to make the model focus on the central area involved applying `CenterAwareOnly` transform *only* to the validation set during evaluation, while the training             process remained unchanged. The details of implementation of  `CenterAwareOnly` transform is given in Appendix.


   - #### Results&Analysis

     |  Without *CenterAwareOnly*    | 75.14%      | 75.14%       | 75.14%       | 75.14%      | 75.14%      |
     | :------------------- | :------ | :------ | :------ | :------ | :------ |
     | *CenterAwareOnly* Size | 200  | 500  | 800  | 1000  | 1500  |
     |   with *CenterAwareOnly*  |  57.95% | 60.41%  | 62.54%  | 63.89% | 65.49%|
     
     It's obvious that model doesn't pay attention to central area or in another way it doesn't know how to pay attention to that specific area.
     We try another advanced areas from a paper.

   - #### Second Method
     Recognizing the limitation of applying CenterCrop only at evaluation, we designed a transform class to explicitly blend the central area with the original picture. The operation is         defined as O=(1−α)⋅R(I)+α⋅R(C(I)), where I is the original image, C(⋅) is center cropping, R(⋅) is global scaling, and α is an integration coefficient. The idea was to emphasize the          central region by overlaying a scaled version of its cropped part.
   - #### Results & Analysis:
     This method did not yield the desired improvement. Its limitation likely stemmed from the simplistic nature of the blending, which might not effectively highlight semantically              relevant features or could introduce artificial artifacts.

2. ### Spectral residual map 

   - #### Method

        This approach utilized spectral residual maps to highlight salient (visually outstanding) regions, aiming to draw the model's attention to key areas of the rock and suppress less           relevant surroundings. The process involved computing a saliency map, generating a binary mask, blurring the background, and then combining the original salient regions with the            blurred background.
   
        The core of the spectral residual method lies in analyzing the image in the **frequency domain**. For an input image $I(x, y)$ in the spatial domain (where $x, y$ are pixel                 coordinates), its Fourier Transform is given by $F(f_x, f_y) = \mathcal{F}\{I(x, y)\}$. We then decompose its spectrum into amplitude and phase components:
         
        $$A(f_x, f_y) = |F(f_x, f_y)|$$
        $$\phi(f_x, f_y) = \text{angle}(F(f_x, f_y))$$
      
        The method then focuses on the **log-amplitude spectrum**, denoted as $L(f_x, f_y) = \log(A(f_x, f_y))$. This step converts multiplicative components into additive ones,                    simplifying analysis.
      
        To identify the unusual or "residual" parts, the average (or smoothed) log-amplitude spectrum, $\bar{L}(f_x, f_y)$, is computed by convolving $L(f_x, f_y)$ with a low-pass filter           (e.g., a Gaussian filter) $h$:
      
        $$\bar{L}(f_x, f_y) = L(f_x, f_y) * h$$
      
        The **spectral residual**, $R(f_x, f_y)$, is then derived by subtracting this average from the original log-amplitude spectrum:
      
        $$R(f_x, f_y) = L(f_x, f_y) - \bar{L}(f_x, f_y)$$
      
        This residual represents the "surprising" or non-redundant information in the frequency domain – components that stand out from the typical spectral patterns.
      
        Finally, the saliency map $S(x, y)$ in the spatial domain is obtained by combining the spectral residual with the original phase information and performing an Inverse Fourier               Transform:
      
        $$S(x, y) = \mathcal{F}^{-1}\{\exp(R(f_x, f_y) + i \phi(f_x, f_y))\}$$
      
        (Here, $\mathcal{F}^{-1}$ denotes the Inverse Fourier Transform). This $S(x, y)$ highlights the visually salient regions by emphasizing frequencies that were considered "residual."
      
        Following the computation of the saliency map $S$, a **binary mask** $M$ is generated by thresholding and blurring $S$ (e.g., $M = \text{Blur}(S > \text{threshold})$). A **blurred          background** $B$ is also created by applying a Gaussian blur to the original image $I$. The final output image $O$ then blends the original image's salient regions (where $M=1$)            with the blurred background (where $M=0$), aiming to emphasize the foreground while suppressing distractions:
      
        $$O = M \cdot I + (I - M) \cdot B$$

   - #### Results
      <div style="display: flex;">
        <div style="flex: 1; padding: 10px;">
          ![Alt text for Image 1](image1.jpg)
          <p style="text-align: center;">Caption for Image 1</p>
        </div>
        <div style="flex: 1; padding: 10px;">
          ![Alt text for Image 2](image2.png)
          <p style="text-align: center;">Caption for Image 2</p>
        </div>
      </div>

     |  Epoch    | 10     | 20       | 21      | 28     | 29      |  30  |
     | :------------------- | :------ | :------ | :------ | :------ | :------ |:------ |
     | Training Accuracy | 78.60%  | 80.79%  | 81.94%  | 79.61% | 81.73% | 83.44%  |
     |  Evaluation Accuracy |  72.62% | 72.59%  | 73.19%  | 71.93% | 72.69%| 73.58%|

## IV Experiment III: Test-Time Augmentation (TTA)

1. ### Method

   Instead of modifying the input images through blending or saliency maps, we implemented Test-Time Augmentation (TTA). This involves applying multiple transformations (resizing, horizontal flipping, center cropping, multi-crop like `FiveCrop`, and color jitter) to each test image. The model then predicts each augmented version, and the final prediction is obtained by averaging the softmax probabilities across all augmentations.

2. #### Results

| Method              | Accuracy |
| :------------------ | :------- |
| Without TTA (Baseline) | 75.04%   |
| With TTA            | **75.71%** |

3. ### Results Summary
TTA provided a modest improvement in accuracy (from 75.04% to 75.71\%). This indicates that aggregating predictions from multiple augmented views does offer some benefit, enhancing the robustness of the final decision. However, the improvement was not substantial. The limitations we identified for this TTA approach include:
* It primarily relies on simple image transformations, which might not be sufficient to address significant domain shifts or complex variations in real-world test data.
* TTA still uses a pre-trained model as its base. If the base model already struggles with certain data characteristics, TTA might not be able to fully compensate.
* The process inherently increases inference time due to multiple forward passes for each test image.

## V Discussion & Summary

### 3.1 Unresolved Challenges and Possible Reasons
Despite various optimization efforts, significant challenges remain, primarily stemming from the model's limited adaptability to unseen data variations and the mismatch between low-level visual saliency and high-level semantic features.

### 3.2 Future Exploration Directions
Future research could explore more advanced Test-Time Adaptation techniques that dynamically adjust model parameters, or investigate ensemble methods combining diverse models.

### 3.3 Main Achievements and Lessons Learned
The project successfully implemented and evaluated several optimization strategies, highlighting the importance of robust data preprocessing and the potential of TTA, while also demonstrating the complexities of aligning low-level image features with high-level semantic understanding.

### 3.4 Project Innovation and Practical Value
Our exploration into blending and saliency-based preprocessing offers insights into attention mechanisms, while the application of TTA enhances the model's practical utility for real-world geological image analysis.

### 3.5 Potential in Future Research and Applications
Data science projects like this hold immense potential for automating geological surveys, enhancing mineral exploration, and aiding educational tools by providing rapid and accurate rock classification capabilities.

# Reference

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale*. In *International Conference on Learning Representations*. https://openreview.net/forum?id=YicbFdNTTy

Wightman, R. (2019). *PyTorch Image Models (timm)* [Computer software]. GitHub repository. https://github.com/rwightman/pytorch-image-models (doi: 10.5281/zenodo.4414861)

X. Hou and L. Zhang, "Saliency Detection: A Spectral Residual Approach," 2007 IEEE Conference on Computer Vision and Pattern Recognition, Minneapolis, MN, USA, 2007, pp. 1-8, doi: 10.1109/CVPR.2007.383267. keywords: {Object detection;Computational modeling;Humans;Visual system;Image analysis;Object recognition;Machine vision;Redundancy;Image coding;Statistical distributions},




