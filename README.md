# DeepFER: Facial Emotion Recognition Using Deep Learning

DeepFER recognises seven emotions (angry, disgust, fear, happy, neutral, sad, surprise) from 48x48 grayscale face images and runs in real time on a laptop webcam. It compares a classic computer vision baseline, a CNN designed from scratch and transfer learning with ResNet18 and EfficientNet-B0, and deploys the best model as a Streamlit app and a live webcam window.

Everything is in one notebook that follows the course ML submission template:
**[DeepFER_Facial_Emotion_Recognition.ipynb](DeepFER_Facial_Emotion_Recognition.ipynb)**

## Repository layout

```
deepfer-emotion-recognition/
├── DeepFER_Facial_Emotion_Recognition.ipynb   the whole project, run top to bottom (trained on a Colab T4 GPU)
├── deepfer.py                                 shared code: loading, cleaning, augmentation, models, training,
│                                              Grad-CAM and face detection (written by the notebook)
├── app/
│   ├── app.py                                 Streamlit app: photo, webcam snapshot, live window, model insights
│   └── webcam.py                              live webcam window (OpenCV)
├── data/
│   ├── face_emotion_images.zip                the 35,887 images in train/ and validation/ folders per emotion
│   └── non_face_images.csv                    58 pictures that are not faces, found while cleaning
├── models/                                    created by the notebook
│   ├── deepfer_cnn.pt                         DeepFER-Net, the custom CNN
│   ├── resnet18.pt                            fine-tuned ResNet18
│   ├── efficientnet_b0.pt                     fine-tuned EfficientNet-B0
│   └── model_card.json                        which networks form the final model, temperature, test scores
├── outputs/
│   ├── figures/                               every chart of the notebook
│   ├── sample_faces/                          28 test faces to try the app (file name = true emotion)
│   ├── demo_frame.png                         a frame with six test faces for the real-time pipeline
│   └── *.csv                                  model comparison, scores per emotion, stress tests, speed, training histories
└── requirements.txt
```

## Data cleaning

| Problem found | Pictures removed |
|---|---|
| Exact duplicates inside the same folder | 1,121 |
| Training copies of test pictures (would leak test answers) | 627 |
| Same picture saved under two different emotions | 148 |
| Not a face: warning signs, "image removed" text, icons | 58 |
| Blank pictures | 14 |

33,919 pictures (94.5%) remain: 22,921 for training, 4,046 for validation and 6,952 for the test (the original validation folder).

## Results (test set, 6,952 faces)

| Model | Accuracy | Macro F1 |
|---|---|---|
| HOG + LBP features, logistic regression (tuned) | 0.452 | 0.385 |
| DeepFER-Net (custom CNN), no augmentation | 0.589 | 0.543 |
| DeepFER-Net, augmented and tuned | 0.682 | 0.661 |
| ResNet18, frozen ImageNet body | 0.448 | 0.351 |
| ResNet18, fine-tuned | 0.687 | 0.667 |
| EfficientNet-B0, fine-tuned | 0.676 | 0.652 |
| **Final: ResNet18 + DeepFER-Net, with mirrored faces** | **0.707** | **0.688** |

The final model's 95% bootstrap interval for accuracy is 0.697 to 0.717. People agree on this kind of data only about 65% of the time (Goodfellow et al., 2013).

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| happy | 0.90 | 0.89 | 0.89 |
| surprise | 0.81 | 0.77 | 0.79 |
| disgust | 0.71 | 0.70 | 0.71 |
| neutral | 0.60 | 0.78 | 0.68 |
| angry | 0.65 | 0.61 | 0.63 |
| sad | 0.58 | 0.58 | 0.58 |
| fear | 0.61 | 0.49 | 0.54 |

Main findings:
- Data augmentation (mirror, rotation, zoom, shift, lighting and cut-out, done on the GPU) was the biggest single gain for the CNN: +9 points of accuracy.
- Frozen ImageNet features are no better than handcrafted HOG features. Fine-tuning the whole network adds more than 20 points.
- Most mistakes are between emotions that look alike: sad read as neutral, fear read as sad or neutral. The same pairs have conflicting labels in the data.
- When the model may say "not sure", it is right 89% of the time on the half of the faces it is most confident about.
- Occlusion tests: the eyes and brows matter most for disgust, angry and fear, the mouth for happy.
- It copes with dark lighting (CLAHE recovers most of the loss on very dark faces) but not with heavy noise, blur or a covered mouth.

## Running the notebook

On **Google Colab** (recommended): open the notebook, choose **Runtime > Change runtime type > T4 GPU** and **Run all**. The notebook downloads the images from this repo, trains everything in about an hour and downloads a zip with the trained models and figures at the end.

Locally (Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook DeepFER_Facial_Emotion_Recognition.ipynb
```

The notebook picks the NVIDIA GPU (CUDA), the Apple GPU (MPS) or the CPU on its own. On my Apple M6 Mac, PyTorch's Apple GPU support froze during training and sometimes changed model weights held in GPU memory, so the final training was done on Colab. The evaluation checks every network against its saved file before using it.

## Running the app (local deployment)

The trained models are in `models/`, so the app works straight after cloning:

```bash
source .venv/bin/activate
streamlit run app/app.py
```

The app opens at http://localhost:8501 with four tabs:
1. **Photo**: upload a photo with one or more faces, or pick a test picture. Every face gets a box, its emotion and confidence, all seven probabilities, the 48x48 crop the model sees and a Grad-CAM heat map. Answers below a confidence threshold are marked as "not sure".
2. **Webcam snapshot**: the same for a picture taken with the webcam.
3. **Live webcam**: opens the live window.
4. **Model insights**: model comparison, confusion matrix, Grad-CAM, occlusion and stress-test results.

The live webcam window can also be started on its own (press q to quit):

```bash
python app/webcam.py
```

Both run on the CPU by default: about 13 ms per face on a laptop, so the webcam runs at more than 20 frames per second.

## Notes and limits

- The model reads facial expressions, not feelings. A polite smile is "happy".
- Emotion recognition is personal data. It should be used with consent, without storing faces, and to support a human decision.
- Fear and sad are the weakest emotions, and noisy or blurred images reduce accuracy a lot. Noise augmentation and a stronger face detector are the next steps.
