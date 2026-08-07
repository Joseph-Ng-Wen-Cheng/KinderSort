"""
sorter.py — Face recognition logic for KinderSort (improved).

This version adds several accuracy and low-end-device optimisations:

- Configurable modes: fast / balanced / accurate / auto that tune detection
  model, encoding model and jitter counts.
- Reference encoding on-disk cache (.kinder_encodings.npz) to avoid re-encoding
  the same reference photos on every run (saves CPU time on repeated runs).
- Uses float32 encodings to reduce memory footprint.
- Adaptive detection: prefer fast HOG detector but fallback to CNN/upsampling
  when no faces are found (configurable by mode).
- Lower default distance threshold to 0.50 to reduce false positives (as in
  project spec), but still configurable on the PhotoSorter class.
- Safer error handling and additional logging when falling back or using cache.

All functions maintain docstrings and use pathlib.Path for file ops.
"""

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
import typing

import numpy as np
from PIL import Image, UnidentifiedImageError

import face_recognition

from utils import (
    build_output_filename,
    collect_event_images,
    is_image_file,
    safe_copy,
)


class PhotoSorter:
    """Encapsulates the full sort pipeline from reference loading to file copying.

    The sorter is configurable to favour accuracy or speed by selecting a
    processing `mode`:
      - 'fast'    : lowest CPU use, minimal jittering, HOG detector only
      - 'balanced': reasonable accuracy / speed tradeoff (default)
      - 'accurate' : higher accuracy, uses larger models and more jitters
      - 'auto'    : choose 'balanced' for low-core / low-RAM machines, otherwise
                    act like 'balanced'

    Caching: reference encodings are cached to a file named
    `.kinder_encodings.npz` inside the reference folder. If reference images
    change (file mtime) the cache is regenerated. This saves a lot of CPU time
    on repeated runs.
    """

    # Default thresholds and sizes tuned for classroom photos
    DISTANCE_THRESHOLD = 0.50  # stricter by default to avoid false matches
    MAX_IMAGE_DIMENSION = 1000  # longest side after resizing for detection

    CACHE_FILENAME = ".kinder_encodings.npz"

    def __init__(
        self,
        reference_folder: Path,
        events_folder: Path,
        output_folder: Path,
        logger: logging.Logger,
        mode: str = "balanced",
    ) -> None:
        """Store folder paths and logger; initialise empty encoding dict.

        Args:
            mode: One of 'fast', 'balanced', 'accurate', or 'auto'.
        """
        self.reference_folder = reference_folder
        self.events_folder = events_folder
        self.output_folder = output_folder
        self.logger = logger
        self._student_encodings: dict[str, np.ndarray] = {}

        self.mode = mode if mode in ("fast", "balanced", "accurate", "auto") else "balanced"
        self._configure_mode()

    def _configure_mode(self) -> None:
        """Set internal parameters according to the selected mode.

        These parameters control the face detection model, the encoding model
        size and how many jitters (re-samplings) are used when computing
        reference encodings.
        """
        # sensible defaults for each mode
        if self.mode == "fast":
            self._detection_model = "hog"
            self._detection_upsample = 0
            self._encoding_model = "small"
            self._num_jitters_ref = 1
            self._num_jitters_detect = 0
        elif self.mode == "accurate":
            self._detection_model = "cnn"
            self._detection_upsample = 1
            self._encoding_model = "large"
            self._num_jitters_ref = 10
            self._num_jitters_detect = 3
        else:  # balanced or auto
            self._detection_model = "hog"
            self._detection_upsample = 0
            self._encoding_model = "large"
            self._num_jitters_ref = 3
            self._num_jitters_detect = 1

        # Auto mode: try to be conservative on very low-resource machines
        if self.mode == "auto":
            try:
                import psutil  # optional, improves auto-detection

                cpu_count = psutil.cpu_count(logical=False) or 1
                mem_gb = psutil.virtual_memory().total / (1024 ** 3)
                if cpu_count <= 2 or mem_gb < 4:
                    # low-end: prefer faster settings
                    self.logger.debug("Auto-mode detected low resources (cpu=%s mem=%.1fGB) — using fast settings", cpu_count, mem_gb)
                    self._detection_model = "hog"
                    self._detection_upsample = 0
                    self._encoding_model = "small"
                    self._num_jitters_ref = 1
                    self._num_jitters_detect = 0
            except Exception:
                # psutil not available — keep balanced defaults
                pass

        self.logger.debug(
            "Mode=%s detection_model=%s detection_upsample=%d encoding_model=%s num_jitters_ref=%d num_jitters_detect=%d",
            self.mode,
            self._detection_model,
            self._detection_upsample,
            self._encoding_model,
            self._num_jitters_ref,
            self._num_jitters_detect,
        )

    # ------------------------------------------------------------------
    # Reference loading (with cache)
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        return self.reference_folder / self.CACHE_FILENAME

    def _load_cache(self) -> dict[str, typing.Any] | None:
        """Load cached encodings if present and valid.

        The cache stores: names, mtimes and encodings (float32). If any file
        changed since the cache was created the cache is considered invalid.
        """
        cache_file = self._cache_path()
        if not cache_file.exists():
            return None

        try:
            npz = np.load(cache_file, allow_pickle=True)
            names = list(npz["names"].astype(str))
            mtimes = list(npz["mtimes"].astype(np.int64))
            encodings = npz["encodings"].astype(np.float32)

            # Verify that files still exist and mtimes match
            for name, mtime in zip(names, mtimes):
                ref_path = self.reference_folder / f"{name}"
                if not ref_path.exists() or int(ref_path.stat().st_mtime_ns) != int(mtime):
                    self.logger.debug("Reference cache invalidated due to change in %s", ref_path)
                    return None

            return {"names": names, "mtimes": mtimes, "encodings": encodings}
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to read reference cache: %s — regenerating", exc)
            return None

    def _save_cache(self, names: list[str], mtimes: list[int], encodings: np.ndarray) -> None:
        """Save the reference encodings to disk (npz compressed).

        Encodings stored as float32 to reduce disk and memory footprint.
        """
        try:
            np.savez_compressed(
                self._cache_path(),
                names=np.array(names, dtype="U"),
                mtimes=np.array(mtimes, dtype=np.int64),
                encodings=np.array(encodings, dtype=np.float32),
            )
            self.logger.debug("Saved reference encoding cache: %s", self._cache_path())
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not save reference cache: %s", exc)

    def load_references(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[str]:
        """Encode every reference photo and store by student name.

        Uses an on-disk cache to avoid re-encoding identical reference photos.

        Returns a list of student names whose reference photo had no detectable
        face (callers will warn the user about these).
        """
        no_face_names: list[str] = []

        reference_images = sorted(
            p for p in self.reference_folder.iterdir() if is_image_file(p)
        )

        if not reference_images:
            self.logger.warning("No reference images found in %s", self.reference_folder)
            return no_face_names

        # Attempt to load cache
        cache = self._load_cache()
        if cache:
            # Rebuild dict from cache
            self._student_encodings = {
                Path(name).stem: enc.astype(np.float32)
                for name, enc in zip(cache["names"], cache["encodings"])
            }
            self.logger.info("Loaded %d student reference(s) from cache", len(self._student_encodings))
            return no_face_names

        total = len(reference_images)
        names: list[str] = []
        mtimes: list[int] = []
        enc_list: list[np.ndarray] = []

        for current, ref_path in enumerate(reference_images, start=1):
            student_name = ref_path.stem
            if progress_callback:
                progress_callback(current, total, student_name)

            try:
                image = face_recognition.load_image_file(str(ref_path))

                # Use the configured detector for locating the face in the reference
                locations = face_recognition.face_locations(
                    image, number_of_times_to_upsample=self._detection_upsample, model=self._detection_model
                )

                encodings = face_recognition.face_encodings(
                    image,
                    known_face_locations=locations if locations else None,
                    num_jitters=self._num_jitters_ref,
                    model=self._encoding_model,
                )

                if not encodings:
                    self.logger.warning(
                        "No face detected in reference photo for %s (%s)",
                        student_name,
                        ref_path.name,
                    )
                    no_face_names.append(student_name)
                    continue

                if len(encodings) > 1:
                    self.logger.warning(
                        "Multiple faces in reference photo for %s — using first face only",
                        student_name,
                    )

                enc = np.asarray(encodings[0], dtype=np.float32)
                self._student_encodings[student_name] = enc
                names.append(ref_path.name)
                mtimes.append(int(ref_path.stat().st_mtime_ns))
                enc_list.append(enc)
                self.logger.info("Loaded reference for %s", student_name)

            except UnidentifiedImageError:
                self.logger.error("Could not read reference image (unrecognised format): %s", ref_path)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Could not read reference photo %s: %s", ref_path.name, exc)

        # Save cache if we successfully encoded at least one reference
        if enc_list:
            try:
                encodings_stack = np.stack(enc_list, axis=0).astype(np.float32)
                self._save_cache(names, mtimes, encodings_stack)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to save encoding cache: %s", exc)

        self.logger.info(
            "Loaded %d student reference(s)", len(self._student_encodings)
        )
        return no_face_names

    # ------------------------------------------------------------------
    # Main sort loop
    # ------------------------------------------------------------------

    def sort_all(
        self,
        progress_callback: Callable[[int, int, str], None],
        cancelled: Callable[[], bool],
    ) -> dict[str, int]:
        """Sort all event photos into per-student output subfolders.

        Processes one image at a time to keep RAM usage low.  For each detected
        face in a photo the nearest student is identified; the photo is copied
        to every matched student folder (allowing group shots).  Photos with no
        match or no face are copied to `_unmatched/`.
        """
        images = collect_event_images(self.events_folder)
        total = len(images)

        counts = {"total": total, "matched": 0, "unmatched": 0, "skipped": 0}

        self.logger.info("Starting sort — %d images found", total)

        # Prepare known encodings array for vectorised distance computation
        names = list(self._student_encodings.keys())
        if names:
            known_encodings = np.stack(list(self._student_encodings.values()), axis=0).astype(np.float32)
        else:
            known_encodings = np.zeros((0, 128), dtype=np.float32)

        for current, (image_path, event_name) in enumerate(images, start=1):
            if cancelled():
                self.logger.info("Sort cancelled by user at image %d/%d", current, total)
                break

            progress_callback(current, total, image_path.name)

            output_filename = build_output_filename(event_name, image_path.name)

            try:
                rgb_image = self._load_and_resize(image_path)
            except UnidentifiedImageError:
                self.logger.warning("Corrupted image, moving to _unmatched: %s", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                continue
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Could not open %s: %s — skipping", image_path.name, exc)
                counts["skipped"] += 1
                continue

            # Face detection with adaptive fallback. Start with a fast HOG pass (or configured model).
            face_locations = []
            try:
                face_locations = face_recognition.face_locations(
                    rgb_image, number_of_times_to_upsample=self._detection_upsample, model=self._detection_model
                )
                # Fallback: if none found and we are allowed more thorough checking,
                # try the other detector or upsampling once.
                if not face_locations and self.mode != "fast":
                    self.logger.debug("No faces found with %s — trying fallback detector/upsample", self._detection_model)
                    # try a single upsample with hog or cnn as the alternative
                    alt_model = "cnn" if self._detection_model == "hog" else "hog"
                    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model=alt_model)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Face detection failed for %s: %s", image_path.name, exc)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                continue

            if not face_locations:
                self.logger.info("No face detected: %s → _unmatched", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                continue

            try:
                # Compute encodings for each detected face. Use configured jittering
                face_encodings = face_recognition.face_encodings(
                    rgb_image, face_locations, num_jitters=self._num_jitters_detect, model=self._encoding_model
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Face encoding failed for %s: %s", image_path.name, exc)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                continue

            if not face_encodings:
                self.logger.info("No face encodings produced: %s → _unmatched", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                continue

            matched_students: set[str] = set()
            for encoding in face_encodings:
                # Convert to float32 and match
                enc32 = np.asarray(encoding, dtype=np.float32)
                match = self._match_face(enc32, names, known_encodings)

                # If a near-miss is found (distance close to threshold) and we are
                # in balanced/accurate mode, re-check with higher-quality encoding
                if match is None and self.mode in ("balanced", "accurate"):
                    try:
                        refined = face_recognition.face_encodings(
                            rgb_image, [face_locations[0]], num_jitters=max(3, self._num_jitters_ref), model="large"
                        )
                        if refined:
                            enc_ref = np.asarray(refined[0], dtype=np.float32)
                            match = self._match_face(enc_ref, names, known_encodings)
                            if match:
                                self.logger.debug("Refined encoding produced match: %s", match)
                    except Exception:
                        # refinement failed — ignore and continue
                        pass

                if match:
                    matched_students.add(match)

            if matched_students:
                for student_name in matched_students:
                    dest_folder = self.output_folder / student_name
                    safe_copy(image_path, dest_folder, output_filename, self.logger)
                    self.logger.info(
                        "Matched %s → %s", image_path.name, student_name
                    )
                counts["matched"] += 1
            else:
                self.logger.info("No match: %s → _unmatched", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1

        self.logger.info(
            "Sort complete — total=%d matched=%d unmatched=%d skipped=%d",
            counts["total"],
            counts["matched"],
            counts["unmatched"],
            counts["skipped"],
        )
        return counts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_and_resize(self, image_path: Path) -> np.ndarray:
        """Open image with Pillow, resize if needed, and return as RGB numpy array.

        Resizing large images to at most MAX_IMAGE_DIMENSION on the longest side
        dramatically reduces face_locations() time on CPU without meaningfully
        reducing recognition accuracy.

        Raises:
            UnidentifiedImageError: If Pillow cannot read the file format.
        """
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            longest = max(width, height)
            if longest > self.MAX_IMAGE_DIMENSION:
                scale = self.MAX_IMAGE_DIMENSION / longest
                new_size = (int(width * scale), int(height * scale))
                # LANCZOS is slower but higher quality — still acceptable when we
                # only resize on large images
                img = img.resize(new_size, Image.LANCZOS)
            return np.array(img)

    def _match_face(self, encoding: np.ndarray, names: list[str], known_encodings: np.ndarray) -> str | None:
        """Find the closest student encoding within DISTANCE_THRESHOLD.

        Uses Euclidean distance (face_recognition.face_distance) over float32
        arrays. The known_encodings array is passed in to avoid rebuilding it
        repeatedly and to keep memory usage predictable.
        """
        if known_encodings.size == 0:
            return None

        # face_recognition.face_distance expects float64 in some versions; cast
        # when calling into it but keep stored arrays as float32 for memory.
        try:
            distances = face_recognition.face_distance(known_encodings.astype(np.float64), encoding.astype(np.float64))
        except Exception:
            # fallback to manual euclidean computation if library call fails
            diffs = known_encodings.astype(np.float32) - encoding.astype(np.float32)
            distances = np.linalg.norm(diffs, axis=1)

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        if best_distance <= self.DISTANCE_THRESHOLD:
            self.logger.debug(
                "Face matched to %s (distance=%.4f)", names[best_idx], best_distance
            )
            return names[best_idx]

        self.logger.debug("No match — best distance=%.4f", best_distance)
        return None


