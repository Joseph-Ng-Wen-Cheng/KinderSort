"""
utils.py — File helpers, naming, and logging for KinderSort.

Improvements in this edit (preprocessing and data handling):
- EXIF-aware image loading and auto-rotation.
- Convert images to RGB numpy arrays sized to a maximum dimension to reduce memory/CPU.
- Pillow-based fallback for image detection when imghdr misses.
- Improved dedup logic and clearer error handling.
- Utility helpers: load_image_for_recognition, open_image_safely.
- APIs used by sorter.py preserved (setup_logger, is_image_file, collect_event_images,
  build_output_filename, safe_copy).
"""

import hashlib
import imghdr
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import ExifTags, Image, UnidentifiedImageError
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Limits for sanitised names
_MAX_FILENAME_LENGTH = 200
_MAX_EVENT_NAME_LENGTH = 80

# Maximum image dimension (largest side) to resize to before face recognition.
# Keeps memory use and CPU lower on low-spec machines while preserving recognisable faces.
MAX_IMAGE_DIMENSION = 1600


def setup_logger(output_folder: Path) -> logging.Logger:
    """Create and return a logger that writes to output_folder/kindersort_log.txt.

    Also attaches a StreamHandler so messages appear in the terminal during
    development. Safe to call multiple times — duplicate handlers are avoided.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / "kindersort_log.txt"
    logger = logging.getLogger("kindersort")
    logger.setLevel(logging.DEBUG)

    # Remove only existing handlers that were previously attached by this module
    # to avoid interfering with other library loggers.
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


def is_image_file(path: Path) -> bool:
    """Return True if path looks like a supported image.

    Fast check: extension-based (case-insensitive). If the extension is
    unrecognised, attempt a lightweight content sniff using imghdr.what().
    If that fails, perform a lightweight Pillow open/verify as a last resort.
    """
    if not path or not path.exists() or not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return True

    # Fallback: sniff file header. imghdr.what accepts a filename string.
    try:
        kind = imghdr.what(str(path))
        if kind is not None:
            kind_ext = f".{kind.lower()}"
            return kind_ext in SUPPORTED_EXTENSIONS
    except Exception:
        pass

    # Final fallback: try Pillow verify (cheap header check). Use try/except to avoid raising.
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def compute_file_hash(path: Path, chunk_size: int = 32768) -> str:
    """Compute MD5 hash of a file in streaming fashion.

    Returns a hex digest string. Uses a moderate chunk size to handle large
    image files without loading them into memory.
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_fragment(name: str, max_len: int) -> str:
    """Sanitise a filename or event name fragment to be filesystem-safe.

    - Strips leading/trailing whitespace
    - Replaces path separators and control characters with underscores
    - Collapses runs of whitespace to single underscore
    - Truncates to max_len
    """
    if not name:
        return ""
    # Replace separators and control chars
    cleaned = name.strip()
    cleaned = cleaned.replace(os.sep, "_")
    cleaned = cleaned.replace("/", "_")
    # collapse any whitespace to single underscore
    cleaned = "_".join(cleaned.split())
    # remove any remaining problematic chars
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in cleaned)
    if len(cleaned) > max_len:
        # keep an end-preserving truncated form (start + last 20 chars)
        keep_start = max_len - 20
        cleaned = cleaned[:keep_start] + "_" + cleaned[-19:]
    return cleaned


def build_output_filename(event_name: str, original_filename: str) -> str:
    """Build a destination filename prefixed with the sanitised event folder name.

    Format: ``{sanitised_event_name}__{sanitised_original_filename}``
    Ensures the returned filename is not excessively long and contains no
    directory separators.
    """
    safe_event = _sanitize_fragment(event_name, _MAX_EVENT_NAME_LENGTH) or "event"
    safe_orig = _sanitize_fragment(Path(original_filename).name, _MAX_FILENAME_LENGTH)
    filename = f"{safe_event}__{safe_orig}"
    # final safety: ensure no separators
    filename = filename.replace(os.sep, "_").replace("/", "_")
    return filename


def ensure_folder(path: Path, exist_ok: bool = True, perms: int | None = None) -> None:
    """Create a folder and optionally set permissions.

    This central helper avoids repeated code and ensures parent folders are
    created with correct behaviour across the codebase.
    """
    path.mkdir(parents=True, exist_ok=exist_ok)
    if perms is not None:
        try:
            path.chmod(perms)
        except Exception:
            # best-effort; ignore permission failures which are platform-specific
            pass


def collect_event_images(events_folder: Path) -> List[Tuple[Path, str]]:
    """Walk immediate subfolders of events_folder and collect image paths.

    Returns a list of (image_path, event_name) tuples where event_name is the
    name of the immediate subfolder containing the image.

    Behaviour improvements:
    - Ignores hidden files and folders (names starting with '.')
    - De-duplicates identical files (same size + MD5 hash) so duplicates across
      different subfolders are not processed multiple times.
    - Falls back to scanning the root of events_folder if no images are found in
      subfolders (flat structure).
    """
    results: List[Tuple[Path, str]] = []
    seen_hashes: Set[str] = set()
    # Map size -> set of observed hashes (optimises collisions-on-size checks)
    size_to_hashes: Dict[int, Set[str]] = {}

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
                # compute hash; small cost but necessary for robust de-duplication
                digest = compute_file_hash(image_path)
                unique_key = f"{size}:{digest}"
                if unique_key in seen_hashes:
                    # duplicate file encountered; skip
                    continue
                seen_hashes.add(unique_key)
                # record per-size mapping (helps future lookups if needed)
                size_to_hashes.setdefault(size, set()).add(digest)
                results.append((image_path, event_name))
            except Exception:
                # If hashing fails, still attempt to include the file (best-effort)
                results.append((image_path, event_name))

    # Scan immediate subdirectories first
    for item in sorted(events_folder.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("."):
            continue
        event_name = _sanitize_fragment(item.name, _MAX_EVENT_NAME_LENGTH) or item.name
        _process_dir(item, event_name)

    # Fallback: if no images found in subfolders, scan root of events_folder directly.
    if not results:
        event_name = _sanitize_fragment(events_folder.name, _MAX_EVENT_NAME_LENGTH) or events_folder.name
        _process_dir(events_folder, event_name)

    return results


def safe_copy(src: Path, dest_folder: Path, filename: str, logger: logging.Logger) -> Path:
    """Copy src to dest_folder/filename safely and atomically.

    - Creates dest_folder if necessary.
    - Writes to a temporary file in the destination folder, then os.replace()
      to atomically move the file into place.
    - If a file with the same name exists, appends _2, _3, … before the extension
      until a free name is found.
    - Retries a few times on transient IO errors.

    Returns the final destination path.
    """
    ensure_folder(dest_folder, exist_ok=True)

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    dest_path = dest_folder / filename

    counter = 2
    # find a non-existing destination path (avoid infinite loop)
    while dest_path.exists():
        dest_path = dest_folder / f"{stem}_{counter}{suffix}"
        counter += 1
        if counter > 1000:
            raise RuntimeError("Unable to find a free filename after 1000 attempts")

    # perform atomic copy via temporary file
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=str(dest_folder)) as tf:
                tmp_path = Path(tf.name)
            # copy to temp path
            shutil.copy2(src, tmp_path)
            # os.replace is atomic on most platforms
            os.replace(tmp_path, dest_path)
            logger.debug("Copied %s → %s", src.name, dest_path)
            return dest_path
        except Exception:
            # cleanup temp file if present
            try:
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < attempts:
                time.sleep(0.1 * attempt)
                continue
            # final failure: fallback to shutil.copy2 (non-atomic)
            try:
                shutil.copy2(src, dest_path)
                logger.debug("Copied (fallback) %s → %s", src.name, dest_path)
                return dest_path
            except Exception:
                logger.exception("Failed to copy %s to %s", src, dest_path)
                raise


# --- Image preprocessing helpers for face recognition ---------------------------------


def open_image_safely(path: Path, logger: Optional[logging.Logger] = None) -> Optional[Image.Image]:
    """Open an image with Pillow and handle common issues.

    Returns a PIL Image (not yet converted) or None if the file cannot be opened.
    """
    try:
        img = Image.open(path)
        # Do not load full image yet; verification ensures header is ok.
        img.verify()
        # Re-open to get a usable Image object (Pillow quirk after verify())
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
    """Apply EXIF orientation to a PIL Image if present.

    Leaves the image unchanged if no EXIF orientation tag is present.
    """
    try:
        exif = img._getexif()
        if not exif:
            return img
        # Find the orientation tag code
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
        # Best-effort: silently continue if EXIF processing fails.
        pass
    return img


def load_image_for_recognition(path: Path, max_dimension: int = MAX_IMAGE_DIMENSION,
                                logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    """Load an image from disk, apply EXIF rotation, convert to RGB and optionally resize.

    Returns an RGB numpy array suitable for face_recognition (H, W, 3) or None on error.

    This function is memory-friendly: it resizes oversized images before converting
    to arrays to reduce memory pressure on low-spec machines.
    """
    img = open_image_safely(path, logger=logger)
    if img is None:
        return None

    try:
        img = _apply_exif_orientation(img)
        # Convert to RGB (face_recognition expects RGB)
        img = img.convert("RGB")

        # Resize if larger than max_dimension
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

