# Histopathologic Cancer Detection with Custom CNN

## Project Overview

This project focuses on developing a **custom Convolutional Neural Network (CNN)** model to detect metastatic cancer in histopathologic images. The goal is to create a binary image classification model that determines whether the center 32x32px region of a histopathologic patch contains cancerous tissue. The project explores the effectiveness of a custom-built CNN compared to pre-trained models like ResNet and EfficientNet.

## Dataset

The dataset used in this project comes from the **Kaggle Histopathologic Cancer Detection competition**, which is based on a modified version of the **PatchCamelyon (PCam)** benchmark dataset. The original PCam dataset contains duplicate images due to probabilistic sampling, but the Kaggle version presented for the competition does not contain duplicates.

### Dataset Overview:
- **Training set**: 220,025 images
- **Test set**: 57,458 images
- **Image format**: `.tif` files
- **Image size**: 96x96px RGB patches
- **Label**: Binary (0 for non-tumor, 1 for tumor)

The task is to identify metastatic tissue in small image patches taken from larger digital pathology scans.

[Histopathologic Cancer Detection on Kaggle](https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview)

## Project Significance

The goal of this project is to create a deep learning model that can assist pathologists in detecting cancerous regions in histopathologic images. Early detection is crucial in cancer diagnosis, and automated detection systems can help speed up the diagnosis process, improve accuracy, and reduce the workload of medical professionals.

## Key Components

### 1. **Custom CNN Architecture**

The architecture includes:
- 4 convolutional layers:
  - Layer 1: 32 filters, 5x5 kernel
  - Layer 2: 128 filters, 3x3 kernel
  - Layer 3: 256 filters, 7x7 kernel
  - Layer 4: 256 filters, 5x5 kernel
- 1 dense layer with 448 units
- Dropout rate: 0.5382
- Initial learning rate: 0.00029326

This custom CNN was compared against pre-trained models like ResNet and EfficientNet to assess performance and understand the trade-offs between custom architectures and pre-established networks.

### 2. **Image Preprocessing**
- **Center cropping**: The critical 32x32 region of each 96x96 image is cropped.
- **Super-resolution enhancement**: The cropped 32x32 region is upscaled to 224x224 using **ESRGAN (Enhanced Super-Resolution GAN)** to improve the quality of input images.

### 3. **Model Training and Hyperparameter Tuning**

Hyperparameter tuning was performed using **Keras Tuner**, focusing on maximizing the **F1-score** through optimization of:
- The number of convolutional layers
- Filter sizes
- Dropout rates
- Dense layer configurations
- Learning rate

### 4. **Threshold Tuning**
Threshold tuning was applied to optimize the **F1-score**, leading to significant improvements in precision and recall.

## Model Performance

After threshold tuning, the final model achieved the following key metrics:
- **Accuracy**: 86.89%
- **Precision**: 84.95%
- **Recall**: 82.20%
- **F1-Score**: 83.55%
- **AUC-ROC**: 0.9359

The optimized model effectively balances precision and recall, making it a reliable tool for cancer detection tasks.

## Future Work

To further improve the model, the following steps could be explored:
- **Ensemble learning**: Combining multiple models (e.g., custom CNN, ResNet, EfficientNet) to improve robustness and accuracy.
- **Advanced data augmentation**: Using more sophisticated data augmentation techniques to enhance model generalization.
- **Transfer learning**: Fine-tuning pre-trained models for better performance on this specific dataset.
- **Model explainability**: Utilizing techniques like Grad-CAM to visualize what the model is focusing on during predictions, increasing interpretability for medical applications.

## Installation and Usage

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- seaborn
- tqdm
- PIL (Pillow)
- TensorFlow Hub (for ESRGAN super-resolution)
- Keras Tuner

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/histopathologic-cancer-detection.git
   cd histopathologic-cancer-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and place the **PatchCamelyon** dataset from [Kaggle's Histopathologic Cancer Detection competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview) in the appropriate folders (`train/`, `test/`, `train_labels.csv`).

### Running the Model

1. **Preprocess images**:
   ```bash
   python preprocess_images.py --input_dir train/ --output_dir train_preprocessed/
   ```

2. **Train the model**:
   ```bash
   python train_model.py
   ```

3. **Make predictions on the test set**:
   ```bash
   python predict.py --model_path final_custom_cnn_model.keras --test_dir test_preprocessed/ --output submission.csv
   ```

### Project Structure

```
.
├── .gitignore
├── Histopathologic_Cancer_Detection.ipynb
├── Histopathologic_Cancer_Detection.md
├── README.md
├── final_custom_cnn_model.keras
├── hyperparameter_tuning
│   └── (tuning results and configuration files)
├── requirements.txt
├── submission.csv
└── visualizations
    ├── 054155c9e3206e565741be06102a1db2c23c31dc.png
    ├── 1a2ab9367eacb8c4e387b197997e9287c74cb934.png
    ├── 1d182790aeaf5f42b89f96ba865ed158b84e2b57.png
    ├── 5a74860a9c5c0a3e2577f793cbf2054cd1aeba08.png
    ├── 7d9deca4ab1d007a08ef89198a362f9826b16c63.png
    ├── CNN_summary.png
    ├── Class_Distribution.png
    ├── Pixel_Intensity_Distribution.png
    ├── Sample_Images.png
    ├── color_distribution.png
    ├── edge.png
    ├── efficientnet_summary.png
    ├── model_comparison_mean_accuracy.png
    ├── optimal_threshold.png
    ├── original_vs_preprocessed_image.png
    ├── prediction_distribution.png
    └── resnet_summary.png
```

## Acknowledgements

- **Kaggle** for hosting the **Histopathologic Cancer Detection** competition and providing the dataset.
- **Keras Tuner** for hyperparameter optimization.
- **ESRGAN Model** for super-resolution image enhancement.

Feel free to reach out for any questions or suggestions! 