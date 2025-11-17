# Facial Recognition using Computer Vision 

This project implements a face recognition system leveraging the power of the face recognition library and OpenCV. It provides functionality for real-time face detection and recognition via a live webcam feed, comparing faces between static images, and utilizing encoded face data for efficient recognition. The system encodes images of known faces stored in a directory, allowing for seamless recognition during video capture. It is a simple yet effective demonstration of how computer vision and deep learning can be integrated to build practical applications.

This project demonstrates a face recognition system using the face_recognition library and OpenCV. The system can:

1. Detect and recognize faces from a live webcam feed.
2. Compare faces between two images.
3. Store and utilize encoded face data for efficient recognition.

## Architecture 

The architecture of this face recognition system is designed as a modular and sequential pipeline that integrates data acquisition, facial detection, deep learning based encoding, identity matching, and real-time visualization. The process begins with the image acquisition layer, where the system either loads stored images from the images/ directory or captures continuous video frames from a webcam. These inputs move into the preprocessing stage, where each frame is converted from OpenCV’s BGR format to RGB, ensuring compatibility with the face_recognition library. The next component is the face detection module, which uses dlib’s HOG-based detector to identify the locations of faces within the frame. Once detected, these face regions are passed to the encoding module, where a ResNet-based deep neural network transforms each face into a 128-dimensional embedding that numerically represents the person's facial features. These encodings are then compared to previously stored known encodings,loaded during initialization through the SimpleFacerec class using Euclidean distance to determine identity matches. The recognition module interprets these comparisons and assigns a name to each face, or labels it as unknown when no match is found. Finally, the processed results feed into the visualization layer, where bounding boxes and labels are drawn on the video frame and displayed in real time. Each component operates in a tightly connected pipeline, enabling the system to perform fast, efficient, and accurate face recognition while maintaining clarity, modularity, and scalability across all files and functions.

![Diagram](arc.jpg)



## Machine Leanring Models 
This face recognition system relies on the deep learning models provided through the face_recognition library, which is built on top of the dlib machine learning framework. These models collectively handle two essential tasks: locating faces within an image and generating numerical encodings that uniquely represent each detected face.

The project uses the HOG (Histogram of Oriented Gradients),  CNN (Convolutional Neural Network) models provided by the face_recognition library, which is built on top of dlib, for face detection and recognition.

### HOG Model 

The first model involved in the pipeline is the HOG-based face detector, which stands for Histogram of Oriented Gradients. This method extracts gradient patterns from an image and learns to identify the structural outlines of a human face, such as the nose, eyes, and jawline. HOG is efficient, lightweight, and CPU-friendly, making it ideal for real-time performance without requiring specialized hardware. It enables quick detection of face locations within a frame and works reliably for general-purpose applications like webcam-based recognition.

![Diagram](hog.jpg)


### CNN

In addition to HOG, the face_recognition library provides a more advanced CNN-based face detector. This model uses a deep convolutional neural network trained specifically for face detection, offering superior accuracy and robustness against variations in lighting, camera angles, and partially occluded faces. While CNN-based detection is more precise, it demands more computational power and performs best when supported by a GPU. For this reason, the project primarily uses the HOG detector, which provides the optimal balance between processing speed and accuracy for real-time applications. However, the system remains compatible with CNN detection for scenarios where accuracy takes priority over speed.

![Diagram](cnn.jpg)


### RESNT'34


Once a face has been detected, the next major component of the model pipeline is the deep learning based face encoding network. The face_recognition library uses a powerful ResNet architecture, specifically a version of ResNet-34, to generate facial encodings. This network was trained on the well-known Labeled Faces in the Wild (LFW) dataset and learns to convert facial images into a compact 128-dimensional feature vector. Each vector acts as a highly discriminative signature of a person's face, capturing subtle structural and geometric features. These encodings are consistent enough to recognize the same person in different lighting conditions, angles, or expressions, yet distinct enough to differentiate between different individuals with high accuracy.

![Diagram](resnet.jpg)


Face recognition in this project is achieved by comparing these encodings using Euclidean distance. When a new face is detected in a video frame or image, the system computes its encoding and measures how closely it matches stored encodings of known individuals. If the distance between two vectors is sufficiently small, the system identifies the person as a match. This combination of deep encoding and vector comparison allows the system to perform fast, accurate recognition in both static and live environments. The entire model workflow from HOG-based detection to ResNet-based encoding forms a robust and efficient pipeline, making this project a practical demonstration of how modern deep learning techniques enable real-time facial recognition on consumer hardware.



## Face Detection:

By default, the face_recognition library uses the HOG-based model, which is fast and efficient for CPU usage. Optionally, it can use a CNN-based model, which is more accurate but requires more computational power and typically a GPU.

## Face Encoding and Recognition:
It uses a deep learning-based ResNet model trained on the Labelled Faces in the Wild (LFW) dataset. This model generates 128-dimensional embeddings (encodings) for each face, which are then compared using distance metrics to identify or verify faces.

The ResNet model ensures robust performance in encoding faces and identifying matches based on their embeddings.

### Contribution 
I conceptualized and developed this project as a Semester Project of Computer Vision.

