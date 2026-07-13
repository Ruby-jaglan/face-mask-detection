# Face Mask Detection

The Face Mask Detection System is a deep learning-based web application developed using Python, TensorFlow/Keras, Flask, OpenCV, NumPy, and Pillow. The system uses a Convolutional Neural Network (CNN) to classify uploaded facial images into two categories: With Mask and Without Mask. It processes the input image, extracts meaningful features, and predicts the result with a confidence score through an easy-to-use web interface.
This project demonstrates the practical application of Artificial Intelligence and Image Processing in solving real-world problems. It provides fast, accurate, and automated detection without requiring manual intervention. The system is designed to be user-friendly, allowing users to upload an image and instantly receive the prediction.
It also serves as an educational project for learning deep learning model training, image classification, and web application deployment using Flask.
Overall, this project highlights how modern AI technologies can be effectively integrated into practical applications to improve public safety, reduce manual effort, and support intelligent decision-making through automated face mask detection.

## Run locally

```powershell
cd "C:\Users\Ruby\OneDrive\Desktop\face mask detection"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd Face-Mask-Detection
py app.py
```

Open `http://127.0.0.1:5000` in the browser.

## Dataset layout

Put labelled images in these two folders before running training:

```
Face-Mask-Detection/dataset/
  with_mask/
  without_mask/
```

The included pre-trained model at `Face-Mask-Detection/model/mask_detector.h5` is used automatically by the web app. The optional training script can be used if you want to replace it with your own model.

## Train the classifier (optional, required for live predictions)

After placing images in both dataset folders, return to the project root and run:

```powershell
py train_model.py
```
