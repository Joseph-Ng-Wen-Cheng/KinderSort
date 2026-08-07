"""
sorter.py — Face recognition logic for KinderSort (memory-optimised).

Enhancements in this version:
- Normalised mode handling with a safe fallback and explicit logging on unknown modes.
- Ambiguity handling improved: ambiguous/near-threshold images are copied to an
  "_uncertain" folder for manual review. They are still counted as unmatched for
  compatibility, but counts include an extra "uncertain" key.
- Confidence scoring logged for accepted matches.
- Ambiguity test combines relative second-best margin and absolute proximity
  heuristics to reduce false positives while surfacing borderline cases.
- Refinement path is used more conservatively and only when the face encoding is near threshold.
- Defensive logging and robust error handling.
- Keeps the memory-efficient single-pass matching approach.
"""
import logging
import gc
import math
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
    """Encapsulates the sort pipeline with memory-conscious matching."""

    DISTANCE_THRESHOLD = 0.50
    MAX_IMAGE_DIMENSION = 1000
    CACHE_FILENAME = ".kinder_encodings.npz"

    # Ambiguity / verification parameters
    SECOND_BEST_RATIO = 1.20       # second_best_dist2 must be >= SECOND_BEST_RATIO * best_dist2
    COSINE_THRESHOLD = 0.35       # minimum cosine similarity to accept (0..1)
    NEAR_THRESHOLD_FACTOR = 0.85  # when best_dist2 is within this * threshold2, treat as "near"
    ABS_MARGIN = 0.02             # absolute squared-distance margin suggesting ambiguity

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

        mode_norm = (mode or "").strip().lower()
        if mode_norm not in ("fast", "balanced", "accurate", "auto"):
            if self.logger:
                self.logger.warning("Unknown mode '%s' — falling back to 'balanced'", mode)
            mode_norm = "balanced"
        self.mode = mode_norm
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
            # balanced and default
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
                    if self.logger:
                        self.logger.debug("Auto-mode low resources — switching to fast settings")
                    self._detection_model = "hog"
                    self._detection_upsample = 0
                    self._encoding_model = "small"
                    self._num_jitters_ref = 1
                    self._num_jitters_detect = 0
                    self._distance_threshold = self.DISTANCE_THRESHOLD
                    self._cosine_threshold = max(0.0, self.COSINE_THRESHOLD - 0.05)
            except Exception:
                # If psutil isn't available, stick with the configured mode
                pass

        if self.logger:
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
                    if self.logger:
                        self.logger.debug("Reference cache invalid because %s changed", ref_path)
                    return None

            return {"names": names, "mtimes": mtimes, "encodings": encodings}
        except Exception as exc:  # noqa: BLE001
            if self.logger:
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
            if self.logger:
                self.logger.debug("Saved reference encoding cache: %s", self._cache_path())
        except Exception as exc:  # noqa: BLE001
            if self.logger:
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
            if self.logger:
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
            if self.logger:
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
                    if self.logger:
                        self.logger.warning(
                            "No face detected in reference photo for %s (%s)",
                            student_name,
                            ref_path.name,
                        )
                    no_face_names.append(student_name)
                    continue

                if len(encodings) > 1:
                    if self.logger:
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

                if self.logger:
                    self.logger.info("Loaded reference for %s", student_name)

                # release large image memory ASAP
                del image
                del locations
                del encodings
                gc.collect()

            except UnidentifiedImageError:
                if self.logger:
                    self.logger.error("Could not read reference image (unrecognised): %s", ref_path)
            except Exception as exc:  # noqa: BLE001
                if self.logger:
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
                if self.logger:
                    self.logger.warning("Failed to save encoding cache: %s", exc)

        if self.logger:
            self.logger.info("Loaded %d student reference(s)", len(self._student_encodings))
        return no_face_names

    def sort_all(
        self,
        progress_callback: Callable[[int, int, str], None],
        cancelled: Callable[[], bool],
    ) -> dict[str, int]:
        images = collect_event_images(self.events_folder)
        total = len(images)

        counts = {"total": total, "matched": 0, "unmatched": 0, "skipped": 0, "uncertain": 0}

        self.logger.info("Starting sort — %d images found", total)

        # Prepare names list once to avoid rebuilding repeatedly
        names = list(self._student_encodings.keys())

        for current, (image_path, event_name) in enumerate(images, start=1):
            if cancelled():
                if self.logger:
                    self.logger.info("Sort cancelled by user at image %d/%d", current, total)
                break

            progress_callback(current, total, image_path.name)
            output_filename = build_output_filename(event_name, image_path.name)

            try:
                rgb_image = self._load_and_resize(image_path)
            except UnidentifiedImageError:
                if self.logger:
                    self.logger.warning("Corrupted image, moving to _unmatched: %s", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                continue
            except Exception as exc:  # noqa: BLE001
                if self.logger:
                    self.logger.error("Could not open %s: %s — skipping", image_path.name, exc)
                counts["skipped"] += 1
                continue

            face_locations = []
            try:
                face_locations = face_recognition.face_locations(
                    rgb_image, number_of_times_to_upsample=self._detection_upsample, model=self._detection_model
                )
                if not face_locations and self.mode != "fast":
                    if self.logger:
                        self.logger.debug("No faces found with %s — trying fallback detector/upsample", self._detection_model)
                    alt_model = "cnn" if self._detection_model == "hog" else "hog"
                    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model=alt_model)
            except Exception as exc:  # noqa: BLE001
                if self.logger:
                    self.logger.error("Face detection failed for %s: %s", image_path.name, exc)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                del rgb_image
                gc.collect()
                continue

            if not face_locations:
                if self.logger:
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
                if self.logger:
                    self.logger.error("Face encoding failed for %s: %s", image_path.name, exc)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                del rgb_image
                gc.collect()
                continue

            if not face_encodings:
                if self.logger:
                    self.logger.info("No face encodings produced: %s → _unmatched", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1
                del rgb_image
                gc.collect()
                continue

            matched_students: set[str] = set()
            uncertain_flag = False

            # Process each face encoding one-by-one to keep peak memory low
            for idx, encoding in enumerate(face_encodings):
                enc32 = np.asarray(encoding, dtype=np.float32)
                match_name, best_dist2, cos_sim, is_near, is_ambiguous = self._match_face_memory_efficient(enc32)

                # If match provided and not ambiguous, accept
                if match_name and not is_ambiguous:
                    matched_students.add(match_name)
                    if self.logger:
                        # compute a human-friendly confidence (0..1)
                        try:
                            distance = math.sqrt(best_dist2)
                            conf = max(0.0, min(1.0, (self._distance_threshold - distance) / max(1e-6, self._distance_threshold)))
                        except Exception:
                            conf = 0.0
                        self.logger.debug("Accepted match %s (conf=%.3f cos=%.3f dist2=%.6f)", match_name, conf, cos_sim, best_dist2)
                elif is_ambiguous:
                    # Mark image as uncertain; we'll copy to _uncertain for manual review
                    uncertain_flag = True
                    if self.logger:
                        self.logger.info("Ambiguous/near-threshold match for %s — marking uncertain (dist2=%.6f cos=%.3f)", image_path.name, best_dist2 or 0.0, cos_sim or 0.0)
                else:
                    # no match for this face; nothing to do
                    if self.logger:
                        self.logger.debug("No match for face %d in %s (dist2=%.6f cos=%.3f)", idx, image_path.name, best_dist2 or float("inf"), cos_sim or 0.0)

                # Optional refinement for near-misses in balanced/accurate modes
                if match_name is None and not is_ambiguous and self.mode in ("balanced", "accurate") and is_near:
                    try:
                        refined = face_recognition.face_encodings(
                            rgb_image, [face_locations[idx]], num_jitters=max(3, self._num_jitters_ref), model="large"
                        )
                        if refined:
                            enc_ref = np.asarray(refined[0], dtype=np.float32)
                            r_name, r_dist2, r_cos, r_is_near, r_is_ambig = self._match_face_memory_efficient(enc_ref)
                            if r_name and not r_is_ambig:
                                matched_students.add(r_name)
                                if self.logger:
                                    distance = math.sqrt(r_dist2) if r_dist2 is not None else 0.0
                                    conf = max(0.0, min(1.0, (self._distance_threshold - distance) / max(1e-6, self._distance_threshold)))
                                    self.logger.debug("Refined encoding produced match %s (conf=%.3f)", r_name, conf)
                            elif r_is_ambig:
                                uncertain_flag = True
                                if self.logger:
                                    self.logger.info("Refined encoding still ambiguous for %s", image_path.name)
                            del enc_ref
                    except Exception:
                        # Ignore refinement errors; refinement is best-effort
                        pass

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
                    if self.logger:
                        self.logger.info("Matched %s → %s", image_path.name, student_name)
                counts["matched"] += 1
            elif uncertain_flag:
                # copy to _uncertain for manual curation and count as unmatched for compatibility
                if self.logger:
                    self.logger.info("Image %s flagged as uncertain → _uncertain", image_path.name)
                safe_copy(image_path, self.output_folder / "_uncertain", output_filename, self.logger)
                counts["uncertain"] += 1
                counts["unmatched"] += 1
            else:
                if self.logger:
                    self.logger.info("No match: %s → _unmatched", image_path.name)
                safe_copy(image_path, self.output_folder / "_unmatched", output_filename, self.logger)
                counts["unmatched"] += 1

        if self.logger:
            self.logger.info(
                "Sort complete — total=%d matched=%d unmatched=%d uncertain=%d skipped=%d",
                counts["total"],
                counts["matched"],
                counts["unmatched"],
                counts["uncertain"],
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

    def _match_face_memory_efficient(self, encoding: np.ndarray) -> tuple[str | None, float | None, float | None, bool, bool]:
        """Memory-efficient nearest-neighbour search with ambiguity and cosine checks.

        Iterates stored encodings one at a time to avoid allocating a large
        distances array. Uses squared-distance comparison to avoid per-candidate
        square-root calls. Performs a second-best margin test and a cosine
        similarity verification to reduce false positives.

        Returns:
            (best_name_or_None, best_dist2_or_None, cos_sim_or_None, is_near, is_ambiguous)
        """
        if not self._student_encodings:
            return None, None, None, False, False

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
            if self.logger:
                self.logger.debug("No match — best dist2=%.6f (threshold2=%.6f)", best_dist2, threshold2)
            return None, best_dist2, None, False, False

        # Determine "near" status (close to threshold)
        is_near = (best_dist2 >= self.NEAR_THRESHOLD_FACTOR * threshold2) or ((threshold2 - best_dist2) < self.ABS_MARGIN)

        # Ambiguity check: ensure the second-best is sufficiently worse than best
        is_ambiguous = False
        if second_best_dist2 < float("inf"):
            if second_best_dist2 < self.SECOND_BEST_RATIO * best_dist2:
                # second best is too close -> ambiguous
                is_ambiguous = True

        # Cosine-similarity verification on the best candidate
        cos_sim = None
        try:
            if best_name is None or best_known_enc is None:
                return None, best_dist2, None, is_near, is_ambiguous
            known_norm = self._student_norms.get(best_name)
            if known_norm is None:
                known_norm = float(np.linalg.norm(best_known_enc.astype(np.float32))) or 1e-10
                self._student_norms[best_name] = known_norm

            target_norm = float(np.linalg.norm(target)) or 1e-10
            cos_sim = float(np.dot(best_known_enc.astype(np.float32), target) / (known_norm * target_norm + 1e-12))

            if cos_sim < self._cosine_threshold:
                # If it's near distance threshold we allow the refinement path in sort_all to try harder.
                if self.logger:
                    self.logger.debug(
                        "Cosine check failed for %s (cos=%.4f dist2=%.6f) — cos threshold=%.3f",
                        best_name,
                        cos_sim,
                        best_dist2,
                        self._cosine_threshold,
                    )
                # If already ambiguous by second-best test, keep ambiguous flag
                # If not ambiguous and not near, reject outright
                if not is_ambiguous and not is_near:
                    return None, best_dist2, cos_sim, is_near, True
                # otherwise mark ambiguous (near/low-cosine)
                is_ambiguous = True
        except Exception:
            # If cosine check fails unexpectedly, fall back to L2 acceptance but mark uncertain
            if self.logger:
                self.logger.debug("Cosine verification failed unexpectedly — treating as ambiguous L2 match for %s", best_name)
            is_ambiguous = True

        # If ambiguous by second-best margin or cosine check, surface as ambiguous
        if is_ambiguous:
            return None, best_dist2, cos_sim, is_near, True

        if self.logger:
            self.logger.debug("Face matched to %s (dist2=%.6f cos=%.4f)", best_name, best_dist2, cos_sim if cos_sim is not None else 0.0)

        return best_name, best_dist2, cos_sim, is_near, False
