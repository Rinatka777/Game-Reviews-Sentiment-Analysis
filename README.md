# Steam Game Reviews Sentiment Analysis

Binary sentiment classification on Steam reviews ([Kaggle — Andrew Mvd](https://www.kaggle.com/datasets/andrewmvd/steam-reviews)).
# Game Reviews Sentiment Analysis (Classical Baseline)

**Task:** Predict Steam review sentiment (recommended vs not) from text.  
**Data:** Kaggle Steam Reviews (subsampled to 200k, balanced 50/50).  
**Models:** TF-IDF + Logistic Regression (word 1–2; alt: char_wb 3–5).

## Results

### Validation
| Model                | n_features | Accuracy | Precision | Recall | F1   | ROC-AUC |
|----------------------|-----------:|---------:|----------:|------:|-----:|-------:|
| TF-IDF word 1–2      | 159,329    | 0.912    | 0.921     | 0.902 | 0.911| 0.971  |
| TF-IDF char_wb 3–5   | 146,966    | 0.904    | 0.912     | 0.894 | 0.903| 0.965  |

### Test (winner: TF-IDF word 1–2)
- Accuracy : **0.912**  
- Precision: **0.922**  
- Recall   : **0.900**  
- F1 score : **0.911**  
- ROC AUC  : **0.971**  
- Confusion matrix: `[[9233, 767], [995, 9005]]`

## Reproduce
```bash
# 1) Build balanced splits
python -m src.data.build_sample --in data/raw/dataset.csv --out data/samples/steam_200k --n 200000 --seed 42

# 2) Train TF-IDF (winner config)
python -m src.models.tfidf.tfidf_train --train data/samples/steam_200k/train.parquet --valid data/samples/steam_200k/valid.parquet --out models/tfidf_word_1_2 --seed 42 --ngram_min 1 --ngram_max 2 --min_df 5 --max_df 0.9 --sublinear_tf --analyzer word --C 1.0 --class_weight balanced

# 3) Evaluate on test
python -m src.models.tfidf_eval --test data/samples/steam_200k/test.parquet --model_dir models/tfidf_word_1_2
