"""
utils.py — File helpers, naming, and logging for KinderSort.

Improvements in this edit (preprocessing and data handling):
- Robust image-file detection: checks extension and falls back to imghdr content sniffing.
- Duplicate image detection when collecting event images (MD5 hash + size) to avoid processing identical files multiple times.
- Hidden files/folders are ignored.
- Sanitise event names and output filenames to be filesystem-safe and length-limited.
- safe_copy uses an atomic replace (write to temporary file then os.replace) and retries to avoid race conditions.
- Added utility helpers: compute_file_hash, ensure_folder.

The APIs used by sorter.py are preserved (setup_logger, is_image_file, collect_event_images,
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

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Limits for sanitised names
_MAX_FILENAME_LENGTH = 200
_MAX_EVENT_NAME_LENGTH = 80


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
    This improves robustness for misnamed files while keeping a cheap
    fallback for the common case.
    """
    if not path or not path.exists() or not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return True

    # Fallback: sniff file header. imghdr.what returns None if the file is
    # not a known image type; this is tolerant and cheap for mislabelled files.
    try:
        kind = imghdr.what(path)
        # imghdr names like 'jpeg', 'png', 'bmp' — map to extensions set
        if kind is not None:
            kind_ext = f".{kind.lower()}"
            return kind_ext in SUPPORTED_EXTENSIONS
    except Exception:
        pass

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


def collect_event_images(events_folder: Path) -> list[tuple[Path, str]]:
    """Walk immediate subfolders of events_folder and collect image paths.

    Returns a list of (image_path, event_name) tuples where event_name is the
    name of the immediate subfolder containing the image.

    Behaviour improvements over the previous version:
    - Ignores hidden files and folders (names starting with '.')
    - De-duplicates identical files (same size + MD5 hash) so duplicates across
      different subfolders are not processed multiple times.
    - Falls back to scanning the root of events_folder if no images are found in
      subfolders (flat structure).
    """
    results: list[tuple[Path, str]] = []
    seen_hashes: set[str] = set()

    if not events_folder.exists() or not events_folder.is_dir():
        return results

    # Helper to process a directory and append files
    def _process_dir(dir_path: Path, event_name: str) -> None:
        for image_path in sorted(dir_path.iterdir()):
            if image_path.name.startswith("."):
                continue
            if not image_path.is_file():
                continue
            if not is_image_file(image_path):
                continue
            try:
                # quick dedupe using size + md5
                size = image_path.stat().st_size
                # combine size into pseudo-key to avoid hashing every file
                size_key = f"{size}:{image_path.name}"
                # compute full hash only if necessary
                digest = compute_file_hash(image_path)
                unique_key = f"{size}:{digest}"
                if unique_key in seen_hashes:
                    continue
                seen_hashes.add(unique_key)
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
        # iterate immediate children (non-recursive) to preserve event grouping
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
        except Exception as exc:
            # cleanup temp file if present
            try:
                if 'tmp_path' in locals() and tmp_path.exists():
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

