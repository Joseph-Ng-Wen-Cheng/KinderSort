"""
utils.py — File helpers, feature extraction, data cleaning, and logging for KinderSort.

This version improves preprocessing and adds feature extraction utilities:
- Optional face encoding extraction using face_recognition (graceful fallback).
- Color histogram feature extraction (RGB).
- Reference database builder with validations (single-face requirement, hashing).
- L2 normalisation and similarity helpers.
- Keeps APIs used by sorter.py: setup_logger, is_image_file, collect_event_images,
  build_output_filename, safe_copy, compute_file_hash, ensure_folder.
"""

from __future__ import annotations

import hashlib
import imghdr
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from PIL import ExifTags, Image, UnidentifiedImageError
import numpy as np

# Optional dependency: face_recognition (wrap imports)
try:
    import face_recognition  # type: ignore

    HAVE_FACE_RECOG = True
except Exception:
    HAVE_FACE_RECOG = False

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Limits for sanitised names
_MAX_FILENAME_LENGTH = 200
_MAX_EVENT_NAME_LENGTH = 80

# Default maximum image dimension for preprocessing
MAX_IMAGE_DIMENSION = 1600


# -------------------------
# Logging and small helpers
# -------------------------
def setup_logger(output_folder: Path) -> logging.Logger:
    """Create and return a logger that writes to output_folder/kindersort_log.txt.

    Safe to call multiple times — duplicate handlers are avoided.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / "kindersort_log.txt"
    logger = logging.getLogger("kindersort")
    logger.setLevel(logging.DEBUG)

    # Remove handlers added by this module previously
    for h in list(logger.handlers):
        logger.removeHandler(h)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def ensure_folder(path: Path, exist_ok: bool = True, perms: int | None = None) -> None:
    """Create a folder and optionally set permissions."""
    path.mkdir(parents=True, exist_ok=exist_ok)
    if perms is not None:
        try:
            path.chmod(perms)
        except Exception:
            pass


def compute_file_hash(path: Path, chunk_size: int = 32768) -> str:
    """Compute MD5 hash of a file in streaming fashion."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_fragment(name: str, max_len: int) -> str:
    """Sanitise a filename or event name fragment to be filesystem-safe."""
    if not name:
        return ""
    cleaned = name.strip()
    cleaned = cleaned.replace(os.sep, "_").replace("/", "_")
    cleaned = "_".join(cleaned.split())
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in cleaned)
    if len(cleaned) > max_len:
        keep_start = max_len - 20
        cleaned = cleaned[:keep_start] + "_" + cleaned[-19:]
    return cleaned


def build_output_filename(event_name: str, original_filename: str) -> str:
    """Build a destination filename prefixed with the sanitised event folder name."""
    safe_event = _sanitize_fragment(event_name, _MAX_EVENT_NAME_LENGTH) or "event"
    safe_orig = _sanitize_fragment(Path(original_filename).name, _MAX_FILENAME_LENGTH)
    filename = f"{safe_event}__{safe_orig}"
    filename = filename.replace(os.sep, "_").replace("/", "_")
    return filename


def is_image_file(path: Path) -> bool:
    """Return True if path looks like a supported image."""
    if not path or not path.exists() or not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return True

    try:
        kind = imghdr.what(str(path))
        if kind is not None:
            kind_ext = f".{kind.lower()}"
            return kind_ext in SUPPORTED_EXTENSIONS
    except Exception:
        pass

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# -------------------------
# Image loading & preproc
# -------------------------
def open_image_safely(path: Path, logger: Optional[logging.Logger] = None) -> Optional[Image.Image]:
    """Open an image with Pillow and handle common issues."""
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        return img
    except UnidentifiedImageError:
        if logger:
            logger.debug("UnidentifiedImageError opening image: %s", path)
        return None
    except Exception:
        if logger:
            logger.exception("Error opening image: %s", path)
        return None


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation to a PIL Image if present."""
    try:
        exif = img._getexif()
        if not exif:
            return img
        orientation_key = None
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orientation_key = tag
                break
        if not orientation_key:
            return img
        orientation = exif.get(orientation_key)
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def load_image_for_recognition(path: Path, max_dimension: int = MAX_IMAGE_DIMENSION,
                                logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    """Load an image, apply EXIF rotation, convert to RGB and optionally resize.

    Returns an RGB numpy array (H, W, 3) or None on error.
    """
    img = open_image_safely(path, logger=logger)
    if img is None:
        return None

    try:
        img = _apply_exif_orientation(img)
        img = img.convert("RGB")
        w, h = img.size
        largest = max(w, h)
        if max_dimension and largest > max_dimension:
            scale = max_dimension / float(largest)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.asarray(img)
        return arr
    except Exception:
        if logger:
            logger.exception("Failed to preprocess image for recognition: %s", path)
        return None


# -------------------------
# Feature extraction
# -------------------------
def extract_color_histogram(img_array: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """Extract a normalized 3D RGB histogram from a numpy image array.

    Returns a 1-D float32 vector.
    """
    if img_array is None:
        return np.zeros(sum(bins), dtype=np.float32)
    # img_array expected shape HxWx3 (RGB)
    hist, _ = np.histogramdd(
        img_array.reshape(-1, 3),
        bins=bins,
        range=[(0, 256), (0, 256), (0, 256)]
    )
    hist = hist.astype(np.float32).flatten()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _normalize_vector(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """L2-normalise a vector, return None for None input."""
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def detect_faces(img_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces and return list of face locations (top, right, bottom, left).

    Requires face_recognition; returns empty list if unavailable.
    """
    if img_array is None:
        return []
    if not HAVE_FACE_RECOG:
        return []
    # face_recognition expects RGB arrays
    try:
        return face_recognition.face_locations(img_array, model="hog")  # or 'cnn' if configured
    except Exception:
        return []


def compute_face_encodings(img_array: np.ndarray, locations: Optional[List[Tuple[int, int, int, int]]] = None) -> List[np.ndarray]:
    """Compute face encodings for the provided image array.

    If face_recognition is not available, returns an empty list.
    """
    if img_array is None or not HAVE_FACE_RECOG:
        return []
    try:
        if locations:
            encs = face_recognition.face_encodings(img_array, known_face_locations=locations)
        else:
            encs = face_recognition.face_encodings(img_array)
        # Convert to numpy arrays and normalise
        encs = [np.asarray(e, dtype=np.float32) for e in encs]
        encs = [(_normalize_vector(e) if e is not None else None) for e in encs]
        return [e for e in encs if e is not None]
    except Exception:
        return []


def extract_image_features(path: Path, logger: Optional[logging.Logger] = None,
                           max_dimension: int = MAX_IMAGE_DIMENSION) -> Dict[str, Any]:
    """Extract features and metadata from a single image file.

    Returned dict contains:
      - path: str
      - size_bytes: int
      - md5: str
      - histogram: list[float]
      - face_count: int
      - face_locations: list[tuple]  (top, right, bottom, left)
      - face_encodings: list[list[float]] (L2-normalised vectors) — empty if not available
    """
    result: Dict[str, Any] = {"path": str(path)}
    try:
        if not path.exists() or not path.is_file():
            result["error"] = "not_found"
            return result

        result["size_bytes"] = path.stat().st_size
        result["md5"] = compute_file_hash(path)

        img = load_image_for_recognition(path, max_dimension=max_dimension, logger=logger)
        if img is None:
            result["error"] = "open_failed"
            return result

        # color histogram
        hist = extract_color_histogram(img)
        result["histogram"] = hist.tolist()

        # face detection & encodings (optional)
        locations = detect_faces(img) if HAVE_FACE_RECOG else []
        result["face_count"] = len(locations)
        result["face_locations"] = locations
        encs = compute_face_encodings(img, locations) if HAVE_FACE_RECOG else []
        result["face_encodings"] = [e.tolist() for e in encs]  # serialisable
        return result

    except Exception:
        if logger:
            logger.exception("Failed to extract features from %s", path)
        result["error"] = "exception"
        return result


# -------------------------
# Reference database builder
# -------------------------
def build_reference_database(ref_folder: Path, logger: Optional[logging.Logger] = None,
                             require_single_face: bool = True) -> Dict[str, Dict[str, Any]]:
    """Scan a folder of reference photos and build a lightweight DB of encodings and metadata.

    Expects one image per student with the filename (without extension) used as the student key.
    Returns a mapping: student_name -> { 'source': path, 'md5':..., 'encoding': [...], 'histogram': [...], 'face_count': int }

    If face_recognition is not installed, encoding lists will be empty. Files without faces are skipped.
    """
    db: Dict[str, Dict[str, Any]] = {}
    if not ref_folder.exists() or not ref_folder.is_dir():
        return db

    for f in sorted(ref_folder.iterdir()):
        if not f.is_file():
            continue
        if f.name.startswith("."):
            continue
        if not is_image_file(f):
            continue
        name = _sanitize_fragment(f.stem, _MAX_FILENAME_LENGTH) or f.stem
        try:
            features = extract_image_features(f, logger=logger)
            if features.get("error"):
                if logger:
                    logger.debug("Reference image skipped (error): %s (%s)", f, features.get("error"))
                continue
            face_count = features.get("face_count", 0)
            if require_single_face and face_count != 1:
                if logger:
                    logger.warning("Reference image %s contains %d faces; expected 1 — skipping.", f.name, face_count)
                # still store metadata to help debugging, but no encoding
                db[name] = {
                    "source": str(f),
                    "md5": features.get("md5"),
                    "face_count": face_count,
                    "encoding": None,
                    "histogram": features.get("histogram"),
                }
                continue

            encs = features.get("face_encodings", [])
            encoding = None
            if encs:
                # use first encoding (common convention for single-face reference)
                encoding = np.asarray(encs[0], dtype=np.float32)
                encoding = _normalize_vector(encoding)
                encoding = encoding.tolist() if encoding is not None else None

            db[name] = {
                "source": str(f),
                "md5": features.get("md5"),
                "face_count": face_count,
                "encoding": encoding,
                "histogram": features.get("histogram"),
            }
        except Exception:
            if logger:
                logger.exception("Failed to process reference file: %s", f)
            continue

    return db


# -------------------------
# Similarity helpers
# -------------------------
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (normalized vectors recommended)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compare_encodings(enc_a: List[float], enc_b: List[float], metric: str = "euclidean") -> float:
    """Compare two encodings (lists) and return a metric value.

    - metric == 'euclidean' returns Euclidean distance (lower = more similar)
    - metric == 'cosine' returns cosine similarity (higher = more similar)
    """
    a = np.asarray(enc_a, dtype=np.float32)
    b = np.asarray(enc_b, dtype=np.float32)
    if metric == "cosine":
        return cosine_similarity(a, b)
    return euclidean_distance(a, b)


# -------------------------
# Image collection & safe copy (dedupe)
# -------------------------
def collect_event_images(events_folder: Path) -> List[Tuple[Path, str]]:
    """Walk immediate subfolders of events_folder and collect image paths.

    Returns a list of (image_path, event_name) tuples where event_name is the
    name of the immediate subfolder containing the image.

    De-duplicates identical files (size + MD5).
    """
    results: List[Tuple[Path, str]] = []
    seen_hashes: Set[str] = set()

    if not events_folder.exists() or not events_folder.is_dir():
        return results

    def _process_dir(dir_path: Path, event_name: str) -> None:
        for image_path in sorted(dir_path.iterdir()):
            if image_path.name.startswith("."):
                continue
            if not image_path.is_file():
                continue
            if not is_image_file(image_path):
                continue
            try:
                size = image_path.stat().st_size
                digest = compute_file_hash(image_path)
                unique_key = f"{size}:{digest}"
                if unique_key in seen_hashes:
                    continue
                seen_hashes.add(unique_key)
                results.append((image_path, event_name))
            except Exception:
                results.append((image_path, event_name))

    for item in sorted(events_folder.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("."):
            continue
        event_name = _sanitize_fragment(item.name, _MAX_EVENT_NAME_LENGTH) or item.name
        _process_dir(item, event_name)

    if not results:
        event_name = _sanitize_fragment(events_folder.name, _MAX_EVENT_NAME_LENGTH) or events_folder.name
        _process_dir(events_folder, event_name)

    return results


def safe_copy(src: Path, dest_folder: Path, filename: str, logger: logging.Logger) -> Path:
    """Copy src to dest_folder/filename safely and atomically."""
    ensure_folder(dest_folder, exist_ok=True)

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    dest_path = dest_folder / filename

    counter = 2
    while dest_path.exists():
        dest_path = dest_folder / f"{stem}_{counter}{suffix}"
        counter += 1
        if counter > 1000:
            raise RuntimeError("Unable to find a free filename after 1000 attempts")

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=str(dest_folder)) as tf:
                tmp_path = Path(tf.name)
            shutil.copy2(src, tmp_path)
            os.replace(tmp_path, dest_path)
            logger.debug("Copied %s → %s", src.name, dest_path)
            return dest_path
        except Exception as exc:
            try:
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < attempts:
                time.sleep(0.1 * attempt)
                continue
            try:
                shutil.copy2(src, dest_path)
                logger.debug("Copied (fallback) %s → %s", src.name, dest_path)
                return dest_path
            except Exception:
                logger.exception("Failed to copy %s to %s", src, dest_path)
                raise


# End of file

