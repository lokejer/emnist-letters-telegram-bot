# EMNIST Letter Classifier Bot

A Telegram bot that identifies handwritten letters using a convolutional neural network (CNN) trained on the EMNIST dataset. Send it a photo of a handwritten letter and it tells you what letter it thinks it is, along with a confidence score.

## What It Does

- **Direct photo** — send a photo to the bot and it replies with the predicted letter and confidence (e.g. *Predicted letter: A (confidence: 87.3%)*)
- **Inline query** — in any chat, type `@botname https://image-url.png` to classify a hosted image without leaving the conversation

## How It Works

### 1. Image Preprocessing

When the bot receives a photo, it runs it through a preprocessing pipeline before feeding it to the model:

- Converts the image to **grayscale** and resizes it to **28×28 pixels** — the exact dimensions the model was trained on
- **Normalises** pixel values from 0–255 to 0–1
- **Auto-inverts** the image if needed — the EMNIST dataset uses white letters on a black background, but real photos are typically the opposite

### 2. Classification

The preprocessed image is passed to a Keras CNN loaded from a saved `.h5` model file. The network outputs a **softmax probability distribution** across 26 classes (A–Z). The class with the highest probability is returned as the prediction, along with its confidence score.

The CNN architecture includes `Conv2D`, `MaxPooling2D`, `BatchNormalization`, and `Dropout` layers, trained on the EMNIST Letters dataset (~145,000 handwritten letter samples).

### 3. Response

The bot formats the result and sends it back to the user via the Telegram Bot API.

## Tech Stack

| Component | Technology |
|---|---|
| Bot framework | `python-telegram-bot` (async, v20+) |
| ML framework | TensorFlow / Keras |
| Image processing | Pillow, NumPy |
| HTTP client | httpx (async, for inline URL fetching) |
| Config | python-dotenv |

## Project Structure

```
cnn-bot/
├── bot.py            # Telegram handlers and entry point
├── classifier.py     # Model loading, preprocessing, inference
├── models/
│   └── best_tuned_model.h5   # Trained Keras CNN
├── requirements.txt
└── .env              # TELEGRAM_API_TOKEN
```

## Running Locally

```bash
pip install -r requirements.txt
python bot.py
```

Requires a `.env` file with your Telegram bot token:
```
TELEGRAM_API_TOKEN=your_token_here
```