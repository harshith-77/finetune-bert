from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from evaluate import load
from transformers import EarlyStoppingCallback

model_name = "bert-base-uncased"

# 1. Dataset Preparation
# 1.1 Loading Dataset
data_files = {"train": "datasets/dataset_train.csv", "test": "datasets/dataset_val.csv"}
dataset = load_dataset("csv", data_files=data_files)
# print(dataset["train"][0])
# Output:
    # Dataset({
    #     features: ['question', 'ES Index', 'label'],
    #     num_rows: 3666
    # })
    # {'question': 'What is the movie name which stars Brad Pitt as a racer?', 'ES Index': 'test_index', 'label': 0}

# 1.2 Dataset Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["question"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# print(tokenized_dataset["train"][0])
# Output:
    # Dataset({
    #     features: ['question', 'ES Index', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
    #     num_rows: 3666
    # })
    # {'question': 'What is the movie name which stars Brad Pitt as a racer', 'ES Index': 'test_index', 'label': 0, 'input_ids': [101, 2054, 2003, 1996, 4224, 2171, 2005, 1996, 2103, 9617, 12380, 5311, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}



# 2. Model Configuration
# 2.1 Model Initialization
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
# print(model.config)

# 2.2 Freezing Models Layers (freeze the lower layers for tasks because domain adaptation isn’t critical in those layers)
    # “When should I freeze layers?” Here’s my rule of thumb:
    # For smaller datasets (e.g., <10k samples), freeze the lower layers.
    # For domain-specific tasks, keep all layers trainable to adapt BERT’s embeddings.

# print(f"Trainable parameters before freezing: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# # Freeze all layers except the classifier
# for param in model.bert.parameters():
#     param.requires_grad = False

# # Keep only the classification head trainable
# for param in model.classifier.parameters():
#     param.requires_grad = True

# print(f"Trainable parameters after freezing: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Output:
    # Trainable parameters before freezing: 109488392
    # Trainable parameters after freezing: 6152


# 3. Training Pipeline
# 3.1 Training Arguments
training_args = TrainingArguments(
    output_dir="./topic_classifier",          # Directory for saving model checkpoints
    eval_strategy="epoch",           # Evaluate at the end of each epoch
    save_strategy="epoch",           # Save model at the end of each epoch
    learning_rate=5e-5,              # Start with a small learning rate
    per_device_train_batch_size=16,  # Batch size per GPU
    per_device_eval_batch_size=16,
    num_train_epochs=100,              # Number of epochs
    weight_decay=0.01,               # Regularization
    save_total_limit=2,              # Limit checkpoints to save space
    load_best_model_at_end=True,     # Automatically load the best checkpoint
    logging_dir="./logs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    fp16=True                        # Enable mixed precision for faster training
)

# 3.2 Custom Metric setup
    # Accuracy alone doesn’t always tell the full story, especially for imbalanced datasets. Prefer using F1-score for a more balanced evaluation.
metric = load("f1")

def compute_metric(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    f1 = metric.compute(predictions=predictions, references=labels, average='macro')
    return f1

# 3.3 Trainer API Initialization and Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metric,  # Custom metric function
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),  # Uncomment if you want to use early stopping
    ]
)
trainer.train()

# 4. Evaluating the Finetuned Model
results = trainer.evaluate(tokenized_dataset["test"])
print("Evaluation Results",results)
# Output: F1 Score = 0.7781321029083879