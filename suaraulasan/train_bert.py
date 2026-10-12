"""Fine-tune IndoBERT di split yang sama dengan baseline, catat ke MLflow.

    uv sync --group bert
    uv run python -m suaraulasan.train_bert --data data/sample.csv
    uv run mlflow ui        # bandingkan dengan run tfidf-logreg
"""

import argparse

import mlflow
import numpy as np
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from suaraulasan.train import LABELS, load, split, validate


def encode(tok, texts, labels, max_len):
    enc = tok(list(texts), truncation=True, max_length=max_len)
    return [
        {"input_ids": i, "attention_mask": m, "labels": y}
        for i, m, y in zip(enc["input_ids"], enc["attention_mask"], labels)
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/reviews.db")
    p.add_argument("--base", default="indobenchmark/indobert-base-p1")
    p.add_argument("--out", default="models/indobert")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=128)
    a = p.parse_args()

    rows = load(a.data)
    validate(rows)
    X_tr, X_te, y_tr, y_te = split(rows)

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.base, num_labels=2, id2label=dict(enumerate(LABELS))
    )
    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=64,
        fp16=True,  # wajib biar muat di GPU 4 GB
        save_strategy="no",
        logging_steps=20,
        report_to="mlflow",
        seed=42,
    )

    # ponytail: tanpa validation set/early stopping; epoch tetap supaya test set tidak dipakai tuning
    with mlflow.start_run(run_name="indobert"):
        mlflow.log_params({"model": a.base, "data": a.data, "max_len": a.max_len})
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=encode(tok, X_tr, y_tr, a.max_len),
            processing_class=tok,
        )
        trainer.train()

        logits = trainer.predict(encode(tok, X_te, y_te, a.max_len)).predictions
        pred = np.argmax(logits, axis=1)
        print(classification_report(y_te, pred, target_names=LABELS))
        f1 = round(f1_score(y_te, pred, average="macro"), 4)
        mlflow.log_metrics({"f1_macro": f1, "n": len(rows)})
        print({"f1_macro": f1, "n": len(rows)})

    trainer.save_model(a.out)
