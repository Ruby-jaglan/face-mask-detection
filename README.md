# Face Mask Detection

Flask application for classifying uploaded face images using an included, pre-trained Keras model.

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
