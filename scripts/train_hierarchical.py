"""Train country classifiers conditioned on the predicted region."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "hierarchical_country"
RESULTS = ROOT / "results"


def load_rows(split: str) -> list[dict[str, Any]]:
    with (PROCESSED / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    train = [
        row for row in load_rows("train")
        if row["region"] != "Unknown" and row["country"] != "Unknown"
    ]
    test = [
        row for row in load_rows("test")
        if row["region"] != "Unknown" and row["country"] != "Unknown"
    ]
    region_models: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    for region in sorted({row["region"] for row in train}):
        train_rows = [row for row in train if row["region"] == region]
        test_rows = [row for row in test if row["region"] == region]
        countries = sorted({row["country"] for row in train_rows})
        region_dir = MODELS / region
        region_dir.mkdir(parents=True, exist_ok=True)
        if len(countries) == 1:
            region_models[region] = {"type": "constant", "country": countries[0]}
            summary[region] = {
                "type": "constant",
                "country": countries[0],
                "train_size": len(train_rows),
                "test_size": len(test_rows),
            }
            continue
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            max_features=300_000, sublinear_tf=True, dtype="float32",
        )
        x_train = vectorizer.fit_transform([row["text"] for row in train_rows])
        x_test = vectorizer.transform([row["text"] for row in test_rows])
        y_train = [row["country"] for row in train_rows]
        y_test = [row["country"] for row in test_rows]
        classifier = LinearSVC(class_weight="balanced")
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        joblib.dump(vectorizer, region_dir / "tfidf_vectorizer.joblib")
        joblib.dump(classifier, region_dir / "linear_svc.joblib")
        region_models[region] = {"type": "model", "countries": countries}
        summary[region] = {
            "type": "model",
            "countries": countries,
            "train_size": len(train_rows),
            "test_size": len(test_rows),
            "accuracy": accuracy_score(y_test, predictions),
            "f1_macro": f1_score(y_test, predictions, average="macro"),
            "test_distribution": dict(Counter(y_test)),
        }
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "index.json").write_text(
        json.dumps(region_models, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hierarchical_country_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
