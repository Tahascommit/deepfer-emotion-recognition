# DeepFER: Facial Emotion Recognition Using Deep Learning

Recognising seven emotions (angry, disgust, fear, happy, neutral, sad, surprise) from 48x48 grayscale face images with a custom CNN and with transfer learning (ResNet18 and EfficientNet-B0), plus a Streamlit app and a live webcam window for real-time use.

Work in progress: the results, charts and trained models are added after the full training run.

## Files

```
deepfer-emotion-recognition/
├── DeepFER_Facial_Emotion_Recognition.ipynb   the whole project in the course template
├── deepfer.py                                 shared code (written by the notebook, used by the app)
├── app/
│   ├── app.py                                 Streamlit app: photo, webcam snapshot, model insights
│   └── webcam.py                              live webcam window (OpenCV)
├── data/
│   ├── face_emotion_images.zip                35,887 face images in train/ and validation/ folders
│   └── non_face_images.csv                    58 pictures that are not faces, found while cleaning
└── requirements.txt
```

## Running the notebook on Colab

Open the notebook in Colab, choose **Runtime > Change runtime type > T4 GPU** and **Run all**. The notebook downloads the images from this repo by itself.
