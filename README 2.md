# Steam Game Reviews Sentiment Analysis

Binary sentiment classification on Steam reviews ([Kaggle — Andrew Mvd](https://www.kaggle.com/datasets/andrewmvd/steam-reviews)).

We train and compare:
1. **TF-IDF + Logistic Regression** — fast classical NLP
2. **DistilBERT Transformer** — fine-tuned for binary sentiment

**Goal:** Predict whether a review recommends a game (positive sentiment) or not (negative sentiment), and serve both models in a FastAPI backend.
