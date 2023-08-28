# Cat-and-Dog-Classification-with-Flask
This repository contains code for building a Convolutional Neural Network (CNN) model using TensorFlow and Keras to classify images of cats and dogs. The model is trained on a dataset of cat and dog images and can predict the class of a given image as either a cat or a dog.

## Prerequisites

Before you run the code, make sure you have the following dependencies installed:

- TensorFlow
- scikit-learn
- seaborn
- matplotlib
- numpy

You can install these dependencies using the following command:

```bash
pip install tensorflow scikit-learn seaborn matplotlib numpy pillow
```

## Dataset

The dataset used for training and validation consists of images of cats and dogs. The dataset directory should be structured as follows:

```
dataset_dir/
│
├── cats/
│   ├── cat_image1.jpg
│   ├── cat_image2.jpg
│   └── ...
│
└── dogs/
    ├── dog_image1.jpg
    ├── dog_image2.jpg
    └── ...
```

Additionally, for testing the model, you should have a separate test dataset with images in a similar structure:

```
test_dataset_dir/
│
├── test_image1.jpg
├── test_image2.jpg
└── ...
```

## Code Overview

The code provided in this repository consists of several sections:

1. **Data Preprocessing**: The dataset is loaded and preprocessed using `ImageDataGenerator`. Data augmentation is performed to increase the diversity of the training dataset.

2. **Model Definition**: A CNN model is defined using the `Sequential` API from Keras. It consists of convolutional layers with max-pooling and dense layers for classification.

3. **Model Compilation and Training**: The model is compiled with the specified optimizer and loss function. It is then trained using the training dataset and validated using the validation dataset.

4. **Model Evaluation**: After training, the model is loaded and evaluated using a separate test dataset. The confusion matrix and classification report are generated to assess the model's performance.

5. **Single Image Prediction**: A trained model is used to predict the class of a single test image. The image is loaded, preprocessed, and the model's prediction is displayed.

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/cat-dog-classifier.git
cd cat-dog-classifier
```

2. Prepare your dataset directories as described above.

3. Run the code using a Python interpreter:

```bash
python cat_dog_classifier.py
```

## Output

The code generates the following outputs:

- Trained model performance metrics (accuracy and loss) during training.
- Confusion matrix heatmap.
- Classification report containing precision, recall, F1-score, and support for each class.
- Predicted class label for a single test image.

