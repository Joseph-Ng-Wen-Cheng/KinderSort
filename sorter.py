"""
sorter.py — Face recognition logic for KinderSort (memory-optimised).

This edit improves matching accuracy while preserving the low-memory
matching approach. Additions:

- Second-best margin check to reject ambiguous nearest neighbours.
- Cosine-similarity verification using cached per-student vector norms.
- Mode-aware threshold tightening for 'accurate' mode.
- Per-reference norm caching computed at load time.
- Keeps memory-conscious iteration over stored encodings.

Other behaviour (modes, caching, adaptive detection) is preserved.
"""

import logging
import gc
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
    """Encapsulates the sort pipeline with memory-conscious matching.

    Configuration and caching are the same as the previous improved version.
    """

    DISTANCE_THRESHOLD = 0.50
    MAX_IMAGE_DIMENSION = 1000
    CACHE_FILENAME = ".kinder_encodings.npz"

    # Ambiguity / verification parameters
    SECOND_BEST_RATIO = 1.20       # second_best_dist2 must be >= SECOND_BEST_RATIO * best_dist2
    COSINE_THRESHOLD = 0.35       # minimum cosine similarity to accept (0..1)
    NEAR_THRESHOLD_FACTOR = 0.85  # when best_dist2 is within this * threshold2, treat as "near"

    def __init__(
        self,
        reference_folder: Path,
        events_folder: Path,
        output_folder: Path,
        logger: logging.Logger,
        mode: str = "balanced",
    ) -> None:
        self.reference_folder = reference_folder
        self.events_folder = events_folder
        self.output_folder = output_folder
        self.logger = logger
        # Keep encodings in a dict of small float32 arrays — avoids large float64
        self._student_encodings: dict[str, np.ndarray] = {}
        # Norm cache for cosine verification (float32)
        self._student_norms: dict[str, float] = {}

        self.mode = mode if mode in ("fast", "balanced", "accurate", "auto") else "balanced"
        self._configure_mode()

    def _configure_mode(self) -> None:
        # Defaults
        if self.mode == "fast":
            self._detection_model = "hog"
            self._detection_upsample = 0
            self._encoding_model = "small"
            self._num_jitters_ref = 1
            self._num_jitters_detect = 0
            self._distance_threshold = self.DISTANCE_THRESHOLD
            self._cosine_threshold = max(0.0, self.COSINE_THRESHOLD - 0.05)
        elif self.mode == "accurate":
            self._detection_model = "cnn"
            self._detection_upsample = 1
            self._encoding_model = "large"
            self._num_jitters_ref = 10
            self._num_jitters_detect = 3
            # be slightly stricter in accurate mode
            self._distance_threshold = max(0.35, self.DISTANCE_THRESHOLD - 0.10)
            self._cosine_threshold = min(1.0, self.COSINE_THRESHOLD + 0.10)
        else:
            self._detection_model = "hog"
            self._detection_upsample = 0
            self._encoding_model = "large"
            self._num_jitters_ref = 3
            self._num_jitters_detect = 1
            self._distance_threshold = self.DISTANCE_THRESHOLD
            self._cosine_threshold = self.COSINE_THRESHOLD

        if self.mode == "auto":
            try:
                import psutil

                cpu_count = psutil.cpu_count(logical=False) or 1
                mem_gb = psutil.virtual_memory().total / (1024 ** 3)
                if cpu_count <= 2 or mem_gb < 4:
                    self.logger.debug("Auto-mode low resources — switching to fast settings")
                    self._detection_model = "hog"
                    self._detection_upsample = 0
                    self._encoding_model = "small"
                    self._num_jitters_ref = 1
                    self._num_jitters_detect = 0
                    self._distance_threshold = self.DISTANCE_THRESHOLD
                    self._cosine_threshold = max(0.0, self.COSINE_THRESHOLD - 0.05)
            except Exception:
                pass

        self.logger.debug(
            "Mode=%s detection_model=%s detection_upsample=%d encoding_model=%s num_jitters_ref=%d num_jitters_detect=%d distance_threshold=%.3f cosine_threshold=%.3f",
            self.mode,
            self._detection_model,
            self._detection_upsample,
            self._encoding_model,
            self._num_jitters_ref,
            self._num_jitters_detect,
            self._distance_threshold,
            self._cosine_threshold,
        )

    def _cache_path(self) -> Path:
        return self.reference_folder / self.CACHE_FILENAME

    def _load_cache(self) -> dict[str, typing.Any] | None:
        cache_file = self._cache_path()
        if not cache_file.exists():
            return None
        try:
            npz = np.load(cache_file, allow_pickle=True)
            names = list(npz["names"].astype(str))
            mtimes = list(npz["mtimes"].astype(np.int64))
            encodings = npz["encodings"].astype(np.float32)

            for name, mtime in zip(names, mtimes):
                ref_path = self.reference_folder / f"{name}"
                if not ref_path.exists() or int(ref_path.stat().st_mtime_ns) != int(mtime):
                    self.logger.debug("Reference cache invalid because %s changed", ref_path)
                    return None

            return {"names": names, "mtimes": mtimes, "encodings": encodings}
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to read reference cache: %s — regenerating", exc)
            return None

    def _save_cache(self, names: list[str], mtimes: list[int], encodings: np.ndarray) -> None:
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

    def _rebuild_norms(self) -> None:
        """Recompute cached vector norms for cosine checks."""
        self._student_norms.clear()
        for name, enc in self._student_encodings.items():
            try:
                n = float(np.linalg.norm(enc.astype(np.float32)))
            except Exception:
                n = float(np.linalg.norm(np.asarray(enc, dtype=np.float32)))
            # avoid zero norm
            self._student_norms[name] = n if n > 0.0 else 1e-10

    def load_references(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[str]:
        no_face_names: list[str] = []

        reference_images = sorted(
            p for p in self.reference_folder.iterdir() if is_image_file(p)
        )

        if not reference_images:
            self.logger.warning("No reference images found in %s", self.reference_folder)
            return no_face_names

        cache = self._load_cache()
        if cache:
            # Rebuild dict from cache without stacking anything in memory beyond each row
            self._student_encodings = {
                Path(name).stem: enc.astype(np.float32)
                for name, enc in zip(cache["names"], cache["encodings"])
            }
            # rebuild norms for cosine tests
            self._rebuild_norms()
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

                # Keep small lists to permit cache saving later
                names.append(ref_path.name)
                mtimes.append(int(ref_path.stat().st_mtime_ns))
                enc_list.append(enc)

                self.logger.info("Loaded reference for %s", student_name)

                # release large image memory ASAP
                del image
                del locations
                del encodings
                gc.collect()

            except UnidentifiedImageError:
                self.logger.error("Could not read reference image (unrecognised): %s", ref_path)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Could not read reference photo %s: %s", ref_path.name, exc)

        # rebuild norms after loading all references
        if self._student_encodings:
            self._rebuild_norms()

        if enc_list:
            try:
                # stack only once for saving; keep main in-memory form as dict of float32 arrays
                encodings_stack = np.stack(enc_list, axis=0).astype(np.float32)
                self._save_cache(names, mtimes, encodings_stack)
                del encodings_stack
                gc.collect()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to save encoding cache: %s", exc)

        self.logger.info("Loaded %d student reference(s)", len(self._student_encodings))
        return no_face_names

    def sort_all(
        self,
        progress_callback: Callable[[int, int, str], None],
        cancelled: Callable[[], bool],
    ) -> dict[str, int]:
        images = collect_event_images(self.events_folder)
        total = len(images)

        counts = {"total": total, "matched": 0, "unmatched": 0, "skipped": 0}

        self.logger.info("Starting sort — %d images found", total)

        # Prepare names list once to avoid rebuilding repeatedly
        names = list(self._student_encodings.keys())

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

            face_locations = []
            try:
                face_locations = face_recognition.face_locations(
                    rgb_image, number_of_times_to_upsample=self._detection_upsample, model=self._detection_model
                )
                if not face_locations and self.mode != "fast":
                    self.logger.debug("No faces found with %s — trying fallback detector/upsample", self._detection_model)
                    alt_model = "cnn" if self._detection_model == "hog" else "hog"
                    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model=alt_model)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Face detection failed for %s: %s", image_path.name, exc)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                del rgb_image
                gc.collect()
                continue

            if not face_locations:
                self.logger.info("No face detected: %s → _unmatched", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                del rgb_image
                gc.collect()
                continue

            try:
                # Compute encodings for each detected face; process them iteratively
                face_encodings = face_recognition.face_encodings(
                    rgb_image, face_locations, num_jitters=self._num_jitters_detect, model=self._encoding_model
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Face encoding failed for %s: %s", image_path.name, exc)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                del rgb_image
                gc.collect()
                continue

            if not face_encodings:
                self.logger.info("No face encodings produced: %s → _unmatched", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                del rgb_image
                gc.collect()
                continue

            matched_students: set[str] = set()

            # Process each face encoding one-by-one to keep peak memory low
            for idx, encoding in enumerate(face_encodings):
                enc32 = np.asarray(encoding, dtype=np.float32)
                match = self._match_face_memory_efficient(enc32)

                # Optional refinement for near-misses in balanced/accurate modes
                if match is None and self.mode in ("balanced", "accurate"):
                    try:
                        refined = face_recognition.face_encodings(
                            rgb_image, [face_locations[idx]], num_jitters=max(3, self._num_jitters_ref), model="large"
                        )
                        if refined:
                            enc_ref = np.asarray(refined[0], dtype=np.float32)
                            match = self._match_face_memory_efficient(enc_ref)
                            if match:
                                self.logger.debug("Refined encoding produced match: %s", match)
                            del enc_ref
                    except Exception:
                        pass

                if match:
                    matched_students.add(match)

                # free per-face memory promptly
                del enc32
                gc.collect()

            # free face encodings and image as we no longer need them
            del face_encodings
            del face_locations
            del rgb_image
            gc.collect()

            if matched_students:
                for student_name in matched_students:
                    dest_folder = self.output_folder / student_name
                    safe_copy(image_path, dest_folder, output_filename, self.logger)
                    self.logger.info("Matched %s → %s", image_path.name, student_name)
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

    def _load_and_resize(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            longest = max(width, height)
            if longest > self.MAX_IMAGE_DIMENSION:
                scale = self.MAX_IMAGE_DIMENSION / longest
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size, Image.LANCZOS)
            arr = np.asarray(img, dtype=np.uint8)
        return arr

    def _match_face_memory_efficient(self, encoding: np.ndarray) -> str | None:
        """Memory-efficient nearest-neighbour search with ambiguity and cosine checks.

        Iterates stored encodings one at a time to avoid allocating a large
        distances array. Uses squared-distance comparison to avoid per-candidate
        square-root calls. Performs a second-best margin test and a cosine
        similarity verification to reduce false positives.

        Returns the best matching student name or None.
        """
        if not self._student_encodings:
            return None

        target = encoding.astype(np.float32)
        best_name: str | None = None
        best_known_enc: np.ndarray | None = None
        best_dist2 = float("inf")
        second_best_dist2 = float("inf")

        threshold2 = float(self._distance_threshold * self._distance_threshold)

        # Iterate over known encodings without creating big temporary arrays
        for name, known_enc in self._student_encodings.items():
            try:
                diff = known_enc - target
                dist2 = float(np.dot(diff, diff))
            except Exception:
                dist2 = float(np.sum((known_enc - target) ** 2))

            # update best and second-best distances
            if dist2 < best_dist2:
                second_best_dist2 = best_dist2
                best_dist2 = dist2
                best_name = name
                best_known_enc = known_enc
            elif dist2 < second_best_dist2:
                second_best_dist2 = dist2

            # early exit if perfect match (very rare)
            if best_dist2 == 0.0:
                break

        # If no candidate within threshold, immediately return
        if best_dist2 > threshold2:
            self.logger.debug("No match — best dist2=%.6f (threshold2=%.6f)", best_dist2, threshold2)
            return None

        # Ambiguity check: ensure the second-best is sufficiently worse than best
        if second_best_dist2 < float("inf"):
            if second_best_dist2 < self.SECOND_BEST_RATIO * best_dist2:
                self.logger.debug(
                    "Ambiguous match rejected for %s (best=%.6f second=%.6f)", best_name, best_dist2, second_best_dist2
                )
                return None

        # Cosine-similarity verification on the best candidate
        try:
            if best_name is None or best_known_enc is None:
                return None
            known_norm = self._student_norms.get(best_name)
            if known_norm is None:
                known_norm = float(np.linalg.norm(best_known_enc.astype(np.float32))) or 1e-10
                self._student_norms[best_name] = known_norm

            target_norm = float(np.linalg.norm(target)) or 1e-10
            cos_sim = float(np.dot(best_known_enc.astype(np.float32), target) / (known_norm * target_norm + 1e-12))

            if cos_sim < self._cosine_threshold:
                # If it's near distance threshold we allow the refinement path in sort_all to try harder.
                self.logger.debug(
                    "Cosine check failed for %s (cos=%.4f dist2=%.6f) — rejecting", best_name, cos_sim, best_dist2
                )
                return None
        except Exception:
            # If cosine check fails unexpectedly, fall back to L2 acceptance (safe default)
            self.logger.debug("Cosine verification failed unexpectedly — accepting L2 match for %s", best_name)

        self.logger.debug("Face matched to %s (dist2=%.6f)", best_name, best_dist2)
        return best_name
