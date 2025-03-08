Overview:

This project focuses on classifying various skin diseases using deep learning techniques. In skinmodel.ipynb , model is created using keras and then it's weights used in web apllication which implemented using flask in app.py


Features:

1.Supports classification of 7 skin diseases.
2.Utilizes Convolutional Neural Networks (CNNs) for image analysis.
3.Pretrained models (ResNet, VGG, etc.) used for feature extraction.
4.User-friendly interface for image upload and classification.
5.Option to visualize model predictions and confidence scores.

Tech Stack:

Backend: Flask
Frontend: HTML, CSS, JavaScript
Machine Learning: TensorFlow/Keras

Requirements:

Flask==1.0.2
Pillow==5.4.1
gevent==1.4.0
gunicorn==19.9.0

How to use?

1.Upload an image of a skin condition.
2.The model processes the image and predicts the disease.
3.Results are displayed with confidence scores.




