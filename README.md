# emnist/letters classifier bot

Telegram bot that classifies handwritten letters using a previous assignment's CNN trained on ~88000 rows of images from the EMNIST/letters dataset.

classify your image via 2 modes:
- **direct photo** — send a photo directly to the bot and it replies with the predicted letter and confidence (e.g. *Predicted letter: Z (confidence: 99.9%)*)
- **inline query** — in any chat, type `@botname https://image-url.png` to classify a hosted image without leaving the conversation

## Workflow

### preprocess image

- convert image to grayscale and resize it to **28×28px**
- normalise pixel values from 0–255 to 0–1
- auto-invert white-background images (training dataset uses black-background images)

### classify

a tuned Keras CNN is loaded from a saved `.h5` model file (see how i trained it [here](https://github.com/lokejer/emnist-letters/blob/main/partA.ipynb)). The network outputs a softmax probability distribution across 26 classes (A–Z), returning the class with the highest probability, along with its confidence score.

the CNN includes `Conv2D`, `MaxPooling2D`, `BatchNormalization`, and `Dropout` layers, with state-of-the-art "block"-style architecture.

### response

the bot formats the result and sends it back to the user via the Telegram Bot API.

## Tech Stack

| component | technology |
|---|---|
| bot framework | `python-telegram-bot` (async, v20+) |
| ML framework | TensorFlow / Keras |
| image processing | Pillow, numpy |
| http client | httpx (async, for inline URL fetching) |
| config | python-dotenv |

## Run It Locally

```bash
pip install -r requirements.txt
python bot.py
```

requires a `.env` file with your Telegram bot token:
```
TELEGRAM_API_TOKEN=your_token_here
```
