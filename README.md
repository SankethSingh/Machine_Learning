#Image Classifier using CIFAR-10 Dataset

This project implements an image classification model using the CIFAR-10 dataset, a well-known dataset of 60,000 32x32 color images in 10 different classes. The model uses TensorFlow and includes a convolutional neural network (CNN) architecture. Additionally, it provides functionality for saving and loading the trained model and predicting new images.

##Features

 - Preprocessing of the CIFAR-10 dataset.

 - CNN architecture for image classification.

 - Visualization of training images with their labels.

 - Saving and loading the trained model.

 - Prediction on new images using the trained model.

##Technologies Used

 - Python

 - TensorFlow/Keras

 - NumPy

 - OpenCV

 - Matplotlib

##Directory Structure

.
├── image_classifier_ml.py  # Main script containing the implementation.
├── car.jpg                # Sample image for prediction.
├── image_classifier.model # Saved trained model (generated after training).

##Installation

Clone the repository:
```bash 
git clone https://github.com/yourusername/image-classifier-cifar10.git
cd image-classifier-cifar10
```
##Install dependencies:
Ensure you have Python installed, then install the required libraries:
```bash
pip install numpy opencv-python matplotlib tensorflow
```
Run the script:
```bash
python image_classifier_ml.py
```
##Usage

Dataset Overview:

 - The CIFAR-10 dataset is automatically downloaded and preprocessed.

 - The dataset includes the following 10 classes:

airplane, car, bird, cat, deer, dog, frog, horse, ship, truck.

##Training the Model:

 - The model is trained on a subset of the CIFAR-10 dataset (20,000 images for training and 4,000 images for testing).

 - The architecture includes convolutional, pooling, flattening, and dense layers.

Visualizing the Data:

 - The first 16 images from the training dataset are displayed with their respective labels for visualization.

##Evaluating the Model:

After training, the model is evaluated on the testing dataset to calculate accuracy and loss.

##Saving and Loading the Model:

The trained model is saved to image_classifier.model.

The saved model can be loaded and used for future predictions without retraining.

##Prediction:

The script includes functionality to predict the class of a new image (car.jpg).

The image is read using OpenCV, processed, and passed through the model for prediction.

##Project Workflow

Data Loading and Preprocessing:

CIFAR-10 dataset is normalized to scale pixel values between 0 and 1.

##Model Architecture:

 - Convolutional layers for feature extraction.

 - MaxPooling layers for down-sampling.

 - Dense layers for classification with a softmax activation function.

###Training:

 - Optimizer: Adam.

 - Loss Function: Sparse Categorical Crossentropy.

 - Metrics: Accuracy.

###Evaluation:

Calculates accuracy and loss on the test set.

###Prediction:

Accepts new images, preprocesses them, and predicts their class using the trained model.

##Results

The model is trained for 10 epochs.

Loss and accuracy are displayed after evaluation.

Sample predictions are made on the provided car.jpg image.

##Future Enhancements

 - Data Augmentation: Improve model robustness by augmenting the dataset (e.g., rotations, flips).

 - Advanced Architectures: Use state-of-the-art architectures like ResNet or EfficientNet.

 - Real-time Predictions: Implement a live prediction pipeline using a webcam or live feed.

 - Deployment: Deploy the model using Flask or FastAPI for serving predictions via a REST API.

##Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

##License

This project is licensed under the MIT License. See the LICENSE file for details.

##Acknowledgments

 - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

 - [TensorFlow Documentation] (https://www.tensorflow.org/)

 - [OpenCV Documentation] (https://docs.opencv.org/)

