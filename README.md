# Automated Wildfire Detection from NASA Satellite Imagery

![GIF of the prediction result](prediction_algeria.gif)

## Project Overview

This project is an end-to-end machine learning pipeline that automatically detects wildfires from satellite imagery. It uses a Convolutional Neural Network (CNN) trained on publicly available data from NASA to identify fire and smoke patterns in satellite tiles of Northern Algeria.

The goal is to demonstrate a real-world application of computer vision for environmental monitoring and disaster response. The entire workflow, from data acquisition to prediction, is automated with Python scripts.

## Key Features

-   **Automated Data Acquisition:** Downloads satellite images and corresponding fire data directly from NASA's public APIs (GIBS and FIRMS).
-   **Data Processing:** Automatically labels and sorts images into `fire` and `no_fire` classes to create a custom dataset.
-   **Deep Learning Model:** Implements a CNN using TensorFlow and Keras to perform binary image classification.
-   **Visualization:** Generates a "before-and-after" GIF to clearly visualize the model's predictions on a test image.

## Technology Stack

-   **Language:** Python 3
-   **Core Libraries:**
    -   TensorFlow / Keras: For building and training the deep learning model.
    -   Pandas: For handling and processing the fire location data.
    -   Requests: For interacting with NASA's APIs.
    -   OpenCV-Python & Pillow: For image processing and manipulation.
    -   Imageio: For creating the output GIF.
    -   Matplotlib: For plotting the model's training history.

## Project Structure
