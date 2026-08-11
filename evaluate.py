# evaluate.py
# Lightweight evaluator to compute per-image matching metrics against a labels CSV.
# Usage: python evaluate.py --refs /path/to/reference --events /path/to/events --labels labels.csv --mode balanced
# labels.csv format: image_relative_path,expected1|expected2  (event-relative paths, e.g., "Sports_Day/IMG_001.jpg,Ali|Siti")

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

from sorter import PhotoSorter
from utils import setup_logger, collect_event_images


def load_labels(labels_csv: Path):
    mapping = {}
    with labels_csv.open("r", encoding="utf-8") as fh:
        rdr = csv.reader(fh)
        for row in rdr:
            if not row or len(row) < 2:
                continue
            img_path = row[0].strip()
            expects = [s.strip() for s in row[1].split("|") if s.strip()]
            mapping[img_path] = set(expects)
    return mapping


def evaluate_once(sorter: PhotoSorter, labels_map: dict, events_folder: Path):
    # We'll iterate images the same way collect_event_images does
    results = {"total": 0, "tp": 0, "fp": 0, "fn": 0}
    per_image = []
    images = collect_event_images(events_folder)
    for image_path, event_name in images:
        rel = str(image_path.relative_to(events_folder))
        expected = labels_map.get(rel, set())
        # Load and encode faces
        try:
            arr = sorter._load_and_resize(image_path)
        except Exception:
            arr = None
        encs = []
        if arr is not None:
            try:
                import face_recognition

                locations = face_recognition.face_locations(arr, number_of_times_to_upsample=sorter._detection_upsample, model=sorter._detection_model)
                encs = face_recognition.face_encodings(arr, locations, num_jitters=sorter._num_jitters_detect, model=sorter._encoding_model)
            except Exception:
                encs = []

        predicted = set()
        for enc in encs:
            try:
                name, _, _, _, _ = sorter._match_face_memory_efficient(enc.astype("float32"))
            except Exception:
                name = None
            if name:
                predicted.add(name)

        # For evaluation we treat a photo as a match if any expected name is in predicted
        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        results["total"] += 1
        results["tp"] += tp
        results["fp"] += fp
        results["fn"] += fn
        per_image.append({"image": rel, "expected": list(expected), "predicted": list(predicted), "tp": tp, "fp": fp, "fn": fn})
    return results, per_image


def print_summary(results):
    total = results["total"]
    tp = results["tp"]
    fp = results["fp"]
    fn = results["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    print(f"images={total} tp={tp} fp={fp} fn={fn} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refs", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--mode", default="balanced", choices=["fast", "balanced", "accurate", "auto"])
    args = p.parse_args()
    refs = Path(args.refs)
    events = Path(args.events)
    labels_csv = Path(args.labels)
    logger = setup_logger(Path("."))
    sorter = PhotoSorter(reference_folder=refs, events_folder=events, output_folder=Path("out_eval"), logger=logger, mode=args.mode)
    no_faces = sorter.load_references()
    if no_faces:
        logger.warning("Some references had no faces: %s", no_faces)
    labels = load_labels(labels_csv)
    results, per_image = evaluate_once(sorter, labels, events)
    print_summary(results)
    # dump per-image results for analysis
    with open("eval_details.json", "w", encoding="utf-8") as fh:
        json.dump(per_image, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
