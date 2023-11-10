# Denoising-images-using-MachineLearning

### Overview

This project focuses on addressing challenges related to Optical Character Recognition (OCR) in the context of printed and scanned documents. The difficulties arise due to various factors such as faded text, low image resolution, and physical damage to the paper. The solution proposed here involves leveraging machine learning, specifically a Random Forest Regressor (RFR), to denoise images before applying OCR. The dataset used in this project is from [Kaggle's Denoising Dirty Documents](https://www.kaggle.com/c/denoising-dirty-documents/data).

### Project Structure

The project is organized into several Python scripts, each serving a specific purpose: <br>

1) **Configuration File:** *config.py* - Stores variables used across multiple scripts. <br>
2) **Image Preprocessing Helper:** *helpers.py* - Defines a helper function for blurring and thresholding documents.
3) **Feature Extraction Script:** *build_features.py* - Extracts features and target values from the dataset.
4) **Model Training Script:** *train_denoiser.py* - Trains a Random Forest Regressor using the extracted features.
5) **Model Application Script:** *denoise_document.py* - Applies the trained model to denoise images in the test set.

### Dataset

The project utilizes Kaggle’s Denoising Dirty Documents dataset, which includes three files: test.zip, train.zip, and train_cleaned.zip. The dataset, although relatively small with only 144 training samples, provides an educational tool for understanding and implementing advanced denoising techniques.

### Denoising Algorithm

The denoising algorithm is inspired by a technique introduced by Colin Priest. It involves applying a 5 x 5 window that slides across both the noisy input image and the target output image. Features are extracted at each window position, consisting of a 25-dimensional feature vector from the noisy input image and a single pixel value from the cleaned image. The goal is to train the RFR to predict the pixel values of the cleaned image based on these features.

Feel free to explore, experiment, and contribute to the project. If you have any questions or feedback, please open an issue or reach out to me or refer [Adrian Rosebrock's website](https://pyimagesearch.com/).

Happy coding!
