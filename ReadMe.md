# 🧠 BERT-based Multi-Class Text Classification

This repository contains a fine-tuning pipeline using `bert-base-uncased` from HuggingFace Transformers on a custom multi-class classification dataset.

---

## 📌 Overview

We fine-tune a pre-trained BERT model to classify natural language questions into one of **8 classes** using a small tabular dataset. This project uses the HuggingFace `transformers`, `datasets`, and `evaluate` libraries for a modern and modular ML training pipeline.

---

## 🗂 Dataset Format

The dataset consists of two CSV files:

* `dataset_train_v2.csv` (Train set)
* `dataset_val_v2.csv` (Validation/Test set)

Each CSV should contain the following columns:

| question                  | ES Index              | label |
| ------------------------- | --------------------- | ----- |
| What is the Zone name...? | jiobeacon\_cell\_wise | 0     |

> `label` is a categorical integer from 0 to 7.

---

## ⚙️ Pipeline Structure

### 1. **Data Loading & Tokenization**

* Loads the CSV files using HuggingFace `datasets`
* Tokenizes the `question` field using `AutoTokenizer` from BERT
* Adds `input_ids`, `attention_mask`, and `token_type_ids` as inputs

### 2. **Model Setup**

* Loads `bert-base-uncased` via `AutoModelForSequenceClassification`
* Sets `num_labels=8` for classification
* Optionally freezes all BERT layers except the classifier head to improve generalization on small datasets

### 3. **Training Configuration**

* Uses `Trainer` API with:

  * `eval_strategy="epoch"`
  * `save_strategy="epoch"`
  * `load_best_model_at_end=True`
  * `early_stopping` with patience of 2 epochs
  * `macro F1-score` as evaluation metric
  * Mixed precision (`fp16=True`) for faster training

### 4. **Evaluation**

* Evaluates the final model using `macro F1-score`
* Prints evaluation results after training

---

## 🏋️‍♂️ Training

Run the script to begin training:

```bash
python finetune.py
```

Make sure the following files are present:

* `datasets/dataset_train_v2.csv`
* `datasets/dataset_val_v2.csv`

---

## 📊 Sample Output

```
Evaluation Results {
  'eval_loss': 0.56,
  'eval_f1': 0.7781,
  'eval_runtime': 12.34,
  ...
}
```

---

## 📌 Dependencies

Install dependencies via pip:

```bash
pip install transformers datasets evaluate
```

To enable GPU + mixed precision (FP16), ensure you have:

* PyTorch with CUDA
* GPU support in `transformers`

---
