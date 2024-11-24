# Facial Recognition using Computer Vision 

This project implements a face recognition system leveraging the power of the face recognition library and OpenCV. It provides functionality for real-time face detection and recognition via a live webcam feed, comparing faces between static images, and utilizing encoded face data for efficient recognition. The system encodes images of known faces stored in a directory, allowing for seamless recognition during video capture. It is a simple yet effective demonstration of how computer vision and deep learning can be integrated to build practical applications.

This project demonstrates a face recognition system using the face_recognition library and OpenCV. The system can:

1. Detect and recognize faces from a live webcam feed.
2. Compare faces between two images.
3. Store and utilize encoded face data for efficient recognition.


The project uses the HOG (Histogram of Oriented Gradients),  CNN (Convolutional Neural Network) models provided by the face_recognition library, which is built on top of dlib, for face detection and recognition.

## Face Detection:

By default, the face_recognition library uses the HOG-based model, which is fast and efficient for CPU usage. Optionally, it can use a CNN-based model, which is more accurate but requires more computational power and typically a GPU.

## Face Encoding and Recognition:
It uses a deep learning-based ResNet model trained on the Labelled Faces in the Wild (LFW) dataset. This model generates 128-dimensional embeddings (encodings) for each face, which are then compared using distance metrics to identify or verify faces.

The ResNet model ensures robust performance in encoding faces and identifying matches based on their embeddings.

