import os
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


def rating_to_sentiment(rating: int) -> str:
    if rating in [1, 2]:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Review" not in df.columns or "Rating" not in df.columns:
        raise ValueError("Input CSV must have 'Review' and 'Rating' columns")
    df["Sentiment"] = df["Rating"].apply(rating_to_sentiment)
    df["Review"] = df["Review"].fillna("").astype(str)
    return df


def prepare_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer):
    label2id = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    dataset = Dataset.from_pandas(df[["Review", "Sentiment"]])

    def tokenize_fn(batch):
        return tokenizer(batch["Review"], truncation=True)

    def map_labels(batch):
        batch["label"] = [label2id[y] for y in batch["Sentiment"]]
        return batch

    dataset = dataset.map(map_labels, batched=True)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["Review", "Sentiment"])

    return dataset, label2id, id2label


def split_dataset(dataset: Dataset, test_size: float = 0.2, seed: int = 42):
    split = dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    return split["train"], split["test"]


def compute_metrics_builder(id2label):
    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)
        labels = pred.label_ids
        report = classification_report(labels, preds, target_names=[id2label[i] for i in range(len(id2label))], output_dict=True, digits=4)
        # Trainer expects a flat dict of scalars to log; return macro F1 and accuracy
        return {
            "accuracy": report["accuracy"],
            "f1_macro": report["macro avg"]["f1-score"],
        }
    return compute_metrics


def ensure_dirs():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def save_eval_reports(y_true, y_pred, id2label):
    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(id2label))))

    with open(os.path.join("reports", "classification_report.txt"), "w") as f:
        f.write(report_txt)
    np.savetxt(os.path.join("reports", "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")


def train_distilbert(csv_path: str = "hotel_reviews.csv", model_name: str = "distilbert-base-uncased", output_dir: str = "artifacts/hf_model", epochs: int = 3, batch_size: int = 16, lr: float = 2e-5, weight_decay: float = 0.01, seed: int = 42):
    ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = load_dataframe(csv_path)
    dataset, label2id, id2label = prepare_dataset(df, tokenizer)
    train_ds, test_ds = split_dataset(dataset, test_size=0.2, seed=seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(SENTIMENT_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=seed,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()

    # Evaluate and save reports
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    save_eval_reports(y_true, y_pred, id2label)

    # Save a pipeline-like folder (tokenizer + model)
    hf_export_dir = os.path.join("artifacts", "hotel_sentiment_distilbert")
    os.makedirs(hf_export_dir, exist_ok=True)
    trainer.model.save_pretrained(hf_export_dir)
    tokenizer.save_pretrained(hf_export_dir)

    print("Saved HF model to:", hf_export_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="hotel_reviews.csv")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    train_distilbert(
        csv_path=args.csv,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


