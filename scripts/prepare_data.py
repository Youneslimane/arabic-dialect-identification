"""Download, normalize, merge, and split Arabic dialect datasets.

The script deliberately uses only the Python standard library. MADAR, NADI, and
QADI files must be downloaded manually when their providers require registration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
IADD_URL = "https://huggingface.co/datasets/evageon/IADD/resolve/main/IADD.json?download=true"

REGION_NAMES = {
    "MGH": "Maghrebi",
    "LEV": "Levantine",
    "EGY": "Egyptian",
    "IRQ": "Iraqi",
    "GLF": "Gulf",
    "MSA": "MSA",
    "GENERAL": "MSA",
}
COUNTRY_NAMES = {
    "MAR": "Morocco", "MOROCCO": "Morocco",
    "TUN": "Tunisia", "TUNISIA": "Tunisia",
    "DZ": "Algeria", "DZA": "Algeria", "ALGERIA": "Algeria",
    "EGY": "Egypt", "EGYPT": "Egypt",
    "IRQ": "Iraq", "IRAQ": "Iraq",
    "SYR": "Syria", "SYRIA": "Syria",
    "JOR": "Jordan", "JORDAN": "Jordan",
    "PSE": "Palestine", "PALESTINE": "Palestine",
    "LBN": "Lebanon", "LEBANON": "Lebanon",
}
REGION_BY_COUNTRY = {
    "Morocco": "Maghrebi", "Tunisia": "Maghrebi", "Algeria": "Maghrebi",
    "Libya": "Maghrebi",
    "Egypt": "Egyptian", "Iraq": "Iraqi",
    "Syria": "Levantine", "Jordan": "Levantine",
    "Palestine": "Levantine", "Lebanon": "Levantine",
    "Qatar": "Gulf", "Saudi Arabia": "Gulf", "United Arab Emirates": "Gulf",
    "Bahrain": "Gulf", "Kuwait": "Gulf", "Oman": "Gulf", "Yemen": "Gulf",
    "Sudan": "Gulf",
}
CITY_INFO = {
    "Rabat": ("Morocco", "Maghrebi"), "Fes": ("Morocco", "Maghrebi"),
    "Tunis": ("Tunisia", "Maghrebi"), "Sfax": ("Tunisia", "Maghrebi"),
    "Algiers": ("Algeria", "Maghrebi"), "Tripoli": ("Libya", "Maghrebi"),
    "Cairo": ("Egypt", "Egyptian"), "Alexandria": ("Egypt", "Egyptian"),
    "Aswan": ("Egypt", "Egyptian"), "Damascus": ("Syria", "Levantine"),
    "Aleppo": ("Syria", "Levantine"), "Beirut": ("Lebanon", "Levantine"),
    "Amman": ("Jordan", "Levantine"), "Jerusalem": ("Palestine", "Levantine"),
    "Baghdad": ("Iraq", "Iraqi"), "Basra": ("Iraq", "Iraqi"),
    "Mosul": ("Iraq", "Iraqi"), "Doha": ("Qatar", "Gulf"),
    "Riyadh": ("Saudi Arabia", "Gulf"), "Jeddah": ("Saudi Arabia", "Gulf"),
    "Muscat": ("Oman", "Gulf"), "Sanaa": ("Yemen", "Gulf"),
    "Benghazi": ("Libya", "Maghrebi"), "Khartoum": ("Sudan", "Gulf"),
    "Salt": ("Jordan", "Levantine"), "MSA": ("Unknown", "MSA"),
}
COUNTRY_CODES = {
    "AE": "United Arab Emirates", "BH": "Bahrain", "DZ": "Algeria",
    "EG": "Egypt", "IQ": "Iraq", "JO": "Jordan", "KW": "Kuwait",
    "LB": "Lebanon", "LY": "Libya", "MA": "Morocco", "OM": "Oman",
    "PL": "Palestine", "PS": "Palestine", "QA": "Qatar", "SA": "Saudi Arabia", "SD": "Sudan",
    "SY": "Syria", "TN": "Tunisia", "YE": "Yemen",
}
TEXT_KEYS = ("sentence", "text", "tweet", "utterance", "source", "arabic")
LABEL_KEYS = ("city", "dialect", "label", "country", "province", "region")
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = URL.sub(" ", text)
    text = text.replace("\u0640", "")
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[\u0000-\u001f\u007f]", " ", text)
    return WHITESPACE.sub(" ", text).strip()


def canonical_region(value: Any, country: str = "") -> str:
    raw = str(value or "").strip().upper()
    if raw in REGION_NAMES:
        return REGION_NAMES[raw]
    return REGION_BY_COUNTRY.get(country, "Unknown")


def canonical_country(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return "Unknown"
    return COUNTRY_NAMES.get(raw.upper(), raw.title() if raw else "Unknown")


def get_field(row: dict[str, Any], names: Iterable[str]) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name in lowered and lowered[name] not in (None, ""):
            return lowered[name]
    return ""


def record(text: Any, region: Any = "", country: Any = "", city: Any = "",
           source: str = "", source_id: Any = "") -> dict[str, str]:
    clean = normalize_text(text)
    city_name = normalize_text(city) or "Unknown"
    city_key = next((name for name in CITY_INFO if name.casefold() == city_name.casefold()), "")
    if city_key:
        city_name = city_key
        city_country, city_region = CITY_INFO[city_key]
        country = country or city_country
        region = region or city_region
    country_name = canonical_country(country)
    region_name = canonical_region(region, country_name)
    return {
        "text": clean,
        "region": region_name,
        "country": country_name,
        "city": city_name,
        "source": source,
        "source_id": str(source_id or ""),
    }


def download_iadd(destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IADD -> {destination}")
    urllib.request.urlretrieve(IADD_URL, destination)


def load_iadd(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    output = []
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            output.append(record(
                get_field(row, ("sentence", "text")),
                get_field(row, ("region",)),
                get_field(row, ("country",)),
                get_field(row, ("city",)),
                "IADD",
                index,
            ))
    return output


def scalar_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples", "records"):
            if isinstance(payload.get(key), list):
                return scalar_rows(payload[key])
    return []


def load_structured(path: Path, source: str) -> list[dict[str, str]]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in handle if line.strip()]
            else:
                rows = scalar_rows(json.load(handle))
    else:
        rows = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = "\t" if "\t" in sample else ","
            rows = list(csv.DictReader(handle, delimiter=delimiter))
    output = []
    for index, row in enumerate(rows):
        text = get_field(row, TEXT_KEYS)
        if text:
            output.append(record(
                text, get_field(row, ("region", "dialect")),
                get_field(row, ("country",)), get_field(row, ("city", "province")),
                source, index,
            ))
    return output


def load_text_file(path: Path, source: str) -> list[dict[str, str]]:
    """Best-effort MADAR-style reader: tab-separated text plus filename labels."""
    output = []
    filename = path.stem
    if ".index" in filename.lower():
        return []
    city = ""
    if ".corpus." in filename:
        city = filename.split(".corpus.", 1)[1]
    city = city.replace("_", " ").title()
    for index, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines()):
        parts = line.split("\t")
        text = parts[-1] if len(parts) > 1 else line
        if (text.strip() and not line.lower().startswith(("id\t", "sentence\t", "text\t",
                                                            "sentid.btec\t", "sent_id\t"))):
            output.append(record(text, "", "", city, source, index))
    return output


def load_archive(path: Path, source: str) -> list[dict[str, str]]:
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if name.endswith("/") or "__macosx" in lower:
                continue
            if source == "NADI" or lower.endswith(("readme.txt", "license.txt", ".py")):
                continue
            if source == "QADI" and "qadi_test.txt" not in lower:
                continue
            if lower.endswith((".txt", ".tsv", ".csv", ".utf8", ".text")):
                data = archive.read(name).decode("utf-8-sig", errors="replace")
                temp = path.parent / f".{path.stem}_{Path(name).name}"
                temp.write_text(data, encoding="utf-8")
                try:
                    if source == "QADI":
                        for index, line in enumerate(data.splitlines()):
                            parts = line.rsplit("\t", 1)
                            if len(parts) == 2:
                                label = parts[1].strip().upper()
                                country = COUNTRY_CODES.get(label, "Unknown")
                                region = "MSA" if label == "MSA" else ""
                                rows.append(record(parts[0], region, country, "", source, index))
                    else:
                        rows.extend(load_text_file(temp, source))
                finally:
                    temp.unlink(missing_ok=True)
    return rows


def load_optional(root: Path, source: str) -> list[dict[str, str]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv"}:
            rows.extend(load_structured(path, source))
        elif path.suffix.lower() in {".txt", ".utf8", ".text"}:
            if source != "QADI" or "train_ids" not in path.name.lower():
                rows.extend(load_text_file(path, source))
        elif path.suffix.lower() == ".zip":
            rows.extend(load_archive(path, source))
    return rows


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    result = []
    for row in rows:
        if not row["text"]:
            continue
        key = (row["text"], row["region"], row["country"], row["city"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["region"], row["country"], row["city"])].append(row)
    result = {"train": [], "validation": [], "test": []}
    for key in sorted(buckets):
        bucket = sorted(
            buckets[key],
            key=lambda row: hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
        )
        n = len(bucket)
        test_n = max(1, round(n * 0.15)) if n >= 3 else 0
        valid_n = max(1, round(n * 0.15)) if n >= 5 else 0
        result["test"].extend(bucket[:test_n])
        result["validation"].extend(bucket[test_n:test_n + valid_n])
        result["train"].extend(bucket[test_n + valid_n:])
    return result


def distribution(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    return {
        "region": dict(sorted(Counter(row["region"] for row in rows).items())),
        "country": dict(sorted(Counter(row["country"] for row in rows).items())),
        "city": dict(sorted(Counter(row["city"] for row in rows).items())),
        "source": dict(sorted(Counter(row["source"] for row in rows).items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--madar-dir", type=Path, default=RAW / "madar")
    parser.add_argument("--nadi-dir", type=Path, default=RAW / "nadi")
    parser.add_argument("--qadi-dir", type=Path, default=RAW / "qadi")
    args = parser.parse_args()
    iadd_path = RAW / "iadd" / "IADD.json"
    download_iadd(iadd_path)
    rows = load_iadd(iadd_path)
    optional = ((args.madar_dir, "MADAR"), (args.nadi_dir, "NADI"))
    external_test_rows = []
    for directory, source in optional:
        loaded = load_optional(directory, source)
        if loaded:
            print(f"Loaded {len(loaded):,} rows from {source}")
        else:
            print(f"SKIPPED {source}: no local files found at {directory}")
        rows.extend(loaded)
    qadi_rows = load_optional(args.qadi_dir, "QADI")
    if qadi_rows:
        external_test_rows.extend(qadi_rows)
        print(f"Loaded {len(qadi_rows):,} QADI labeled test rows (kept outside training splits)")
    else:
        print(f"SKIPPED QADI: no labeled text file found at {args.qadi_dir}")
    before = len(rows)
    preprocessed = len([row for row in rows if row["text"]])
    rows = deduplicate(rows)
    splits = split_rows(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROCESSED / "all.jsonl", rows)
    for name, split in splits.items():
        write_jsonl(PROCESSED / f"{name}.jsonl", split)
    if external_test_rows:
        write_jsonl(PROCESSED / "qadi_test.jsonl", external_test_rows)
    report = {
        "raw_rows": before,
        "after_preprocessing_rows": preprocessed,
        "deduplicated_rows": len(rows),
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "overall_distribution": distribution(rows),
        "split_distributions": {name: distribution(split) for name, split in splits.items()},
        "external_test": {
            "qadi_labeled_rows": len(external_test_rows),
            "distribution": distribution(external_test_rows) if external_test_rows else {},
        },
        "preprocessing": ["URLs removed", "Arabic diacritics and tatweel removed",
                          "Alef variants normalized", "Ya and ta marbuta normalized",
                          "whitespace/control characters normalized"],
    }
    (PROCESSED / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not rows:
        print("ERROR: no records were loaded.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
