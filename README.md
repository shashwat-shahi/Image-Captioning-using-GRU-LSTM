# 🖼️ Image Captioning using Deep Learning (GRU & LSTM)

This project implements an image captioning system using deep learning architectures—GRU (Gated Recurrent Units) and LSTM (Long Short-Term Memory). It includes exploratory data analysis (EDA), feature extraction from images, and sequence modeling for generating natural language captions from image inputs.

---

## 📂 Project Structure

- `eda.ipynb`: Performs exploratory data analysis on the image-caption dataset (e.g., distribution of captions, length, vocabulary size).
- `feature_visualization.ipynb`: Extracts visual features using a CNN (e.g., VGG16, InceptionV3) and visualizes the image embeddings.
- `GRU_IMG_Captioning.ipynb`: Implements and trains the GRU-based image captioning model.
- `LSTM_IMG_Captioning.ipynb`: Implements and trains the LSTM-based image captioning model for comparison.

---

## 🧠 Model Overview

The core idea of this project is to use:

- A **CNN encoder** to extract image features.
- A **RNN decoder** (either GRU or LSTM) to generate captions word-by-word based on those features.
- **Tokenization and padding** to prepare the caption data for sequence modeling.
- **Beam search or greedy decoding** to generate predictions during inference.

---

## 🔧 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- Python 3.8+
- TensorFlow or PyTorch (depending on implementation)
- NumPy, Pandas, Matplotlib, Seaborn
- NLTK / SpaCy for text preprocessing
- OpenCV / PIL for image preprocessing

---

## 🚀 Running the Project

1. Clone this repository

2. Prepare your dataset (e.g., [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)) and place it in a `data/` folder.

3. Run the notebooks in this order:
   - `eda.ipynb`
   - `feature_visualization.ipynb`
   - `GRU_IMG_Captioning.ipynb` or `LSTM_IMG_Captioning.ipynb`

---

## 📊 Evaluation

Models are evaluated based on:

- **BLEU Score**: Measures how close the generated captions are to the ground truth.
- **Loss Curves**: Helps visualize model performance during training.
- **Sample Predictions**: Visual inspection of generated captions.

---

## 📌 Future Work

- Implement attention mechanisms to improve caption generation.
- Experiment with different CNN encoders (e.g., ResNet, EfficientNet).
- Optimize model performance with transformer-based architectures like ViT + GPT.
- Deploy as a web application for demo purposes.

---

## 📸 Sample Output

> 🖼️ _Image_ → "A group of people playing frisbee in the park"
