"""Prepare Region/Country tasks and train TF-IDF baseline classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SPLITS = ("train", "validation", "test")


def load_rows(split: str) -> list[dict[str, Any]]:
    path = PROCESSED / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_rows(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    label_key = task
    return [
        row for row in rows
        if row.get("text") and row.get(label_key) and row[label_key] != "Unknown"
    ]


def write_task_splits(tasks: tuple[str, ...]) -> dict[str, dict[str, int]]:
    sizes: dict[str, dict[str, int]] = {}
    for task in tasks:
        sizes[task] = {}
        for split in SPLITS:
            rows = task_rows(load_rows(split), task)
            output = PROCESSED / f"{task}_{split}.jsonl"
            with output.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            sizes[task][split] = len(rows)
    return sizes


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def train_task(task: str, split_sizes: dict[str, int]) -> dict[str, Any]:
    train_rows = task_rows(load_rows("train"), task)
    validation_rows = task_rows(load_rows("validation"), task)
    test_rows = task_rows(load_rows("test"), task)
    x_train = [row["text"] for row in train_rows]
    x_validation = [row["text"] for row in validation_rows]
    x_test = [row["text"] for row in test_rows]
    y_train = [row[task] for row in train_rows]
    y_validation = [row[task] for row in validation_rows]
    y_test = [row[task] for row in test_rows]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
        dtype="float32",
    )
    x_train_matrix = vectorizer.fit_transform(x_train)
    x_validation_matrix = vectorizer.transform(x_validation)
    x_test_matrix = vectorizer.transform(x_test)
    task_results: dict[str, Any] = {"split_sizes": split_sizes, "models": {}}
    task_model_dir = MODELS / task
    task_model_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, task_model_dir / "tfidf_vectorizer.joblib")

    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=100, class_weight="balanced", solver="saga", n_jobs=-1
        ),
        "linear_svc": LinearSVC(class_weight="balanced"),
    }
    for name, classifier in classifiers.items():
        classifier.fit(x_train_matrix, y_train)
        validation_pred = classifier.predict(x_validation_matrix)
        test_pred = classifier.predict(x_test_matrix)
        model_results = {
            "validation": metrics(y_validation, validation_pred),
            "test": metrics(y_test, test_pred),
            "classes": list(classifier.classes_),
        }
        task_results["models"][name] = model_results
        joblib.dump(classifier, task_model_dir / f"{name}.joblib")

        matrix = confusion_matrix(y_test, test_pred, labels=classifier.classes_)
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=classifier.classes_
        )
        figure, axis = plt.subplots(figsize=(9, 7))
        display.plot(ax=axis, xticks_rotation=45, cmap="Blues", colorbar=False)
        axis.set_title(f"{task.title()} classification - {name}")
        figure.tight_layout()
        figure.savefig(
            RESULTS / f"{task}_{name}_confusion_matrix.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)
    return task_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=("region", "country"),
                        default=("region", "country"))
    args = parser.parse_args()
    split_sizes = write_task_splits(tuple(args.tasks))
    all_results = {
        task: train_task(task, split_sizes[task]) for task in args.tasks
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "baseline_metrics.json").write_text(
        json.dumps({"split_sizes": split_sizes, "tasks": all_results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "split_sizes": split_sizes,
        "test_metrics": {
            task: {
                name: {
                    key: model["test"][key] for key in ("accuracy", "f1_macro", "f1_weighted")
                }
                for name, model in result["models"].items()
            }
            for task, result in all_results.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
