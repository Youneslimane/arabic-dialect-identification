"""Predict region and country with the hierarchical baseline models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

try:
    from prepare_data import normalize_text
except ModuleNotFoundError:
    from scripts.prepare_data import normalize_text


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def load_classifier(task: str) -> tuple[Any, Any]:
    directory = MODELS / task
    return (
        joblib.load(directory / "tfidf_vectorizer.joblib"),
        joblib.load(directory / "linear_svc.joblib"),
    )


def predict(text: str) -> dict[str, Any]:
    clean_text = normalize_text(text)
    if not clean_text:
        raise ValueError("The input text is empty after preprocessing.")
    region_vectorizer, region_classifier = load_classifier("region")
    region_features = region_vectorizer.transform([clean_text])
    region = str(region_classifier.predict(region_features)[0])

    country_index = MODELS / "hierarchical_country" / "index.json"
    index = json.loads(country_index.read_text(encoding="utf-8"))
    country_info = index.get(region)
    if country_info is None:
        raise RuntimeError(f"No hierarchical country model is available for {region}.")
    if country_info["type"] == "constant":
        country = country_info["country"]
    else:
        directory = MODELS / "hierarchical_country" / region
        vectorizer = joblib.load(directory / "tfidf_vectorizer.joblib")
        classifier = joblib.load(directory / "linear_svc.joblib")
        country = str(classifier.predict(vectorizer.transform([clean_text]))[0])
    return {"text": clean_text, "region": region, "country": country}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Predict Arabic region and country.")
    parser.add_argument("--text", help="Arabic text to classify.")
    args = parser.parse_args()
    text = args.text or input("Enter Arabic text: ")
    print(predict(text))


if __name__ == "__main__":
    main()
