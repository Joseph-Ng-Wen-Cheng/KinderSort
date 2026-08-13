import gc
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Callable, Optional, Literal

import numpy as np
from ai_backends import BackendManager
from report_generator import generate_student_grid_report

try:
    import face_recognition
except ImportError:
    face_recognition = None


class PhotoSorter:
    DISTANCE_THRESHOLD = 0.6

    def __init__(
        self,
        ref_dir: Path | str,
        events_dir: Path | str,
        output_dir: Path | str,
        logger: Optional[logging.Logger] = None,
        mode: str = "auto",
        ai_backend: str = "auto",
        nvidia_api_key: Optional[str] = None,
        num_jitters_match: int = 1,
        gc_interval: int = 20,
    ):
        self.ref_dir = Path(ref_dir)
        self.events_dir = Path(events_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger("KinderSort")
        self.mode = mode
        self.ai_backend = ai_backend
        self.nvidia_api_key = nvidia_api_key
        self.num_jitters_match = num_jitters_match
        self.gc_interval = gc_interval

        self._student_names: list[str] = []
        self._student_encodings: dict[str, list[np.ndarray]] = {}
        self._ref_targets: list[tuple[str, Path]] = []

        self.backend_mgr = BackendManager(
            preferred_backend=self.ai_backend,
            nvidia_api_key=self.nvidia_api_key,
            logger=self.logger,
        )

        self.load_references()

    def load_references(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs,
    ) -> None:
        """Public interface for main.py to trigger reference loading and generate Word report."""
        self._student_names.clear()
        self._student_encodings.clear()
        self._ref_targets.clear()
        self._load_reference_photos(progress_callback=progress_callback, **kwargs)

        # Generate single-page Word grid report of all loaded reference images
        if self._ref_targets:
            report_path = self.output_dir / "Student_Roster_Report.docx"
            generate_student_grid_report(self._ref_targets, report_path, self.logger)

    def _load_reference_photos(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs,
    ) -> None:
        """Load reference face encodings from individual files or subfolders."""
        if not self.ref_dir.exists():
            self.logger.warning(f"Reference folder {self.ref_dir} does not exist.")
            return

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        # Collect reference targets (both direct image files and subdirectories)
        for item in self.ref_dir.iterdir():
            if item.is_file() and item.suffix.lower() in valid_exts:
                self._ref_targets.append((item.stem, item))
            elif item.is_dir():
                for sub_item in item.rglob("*"):
                    if sub_item.is_file() and sub_item.suffix.lower() in valid_exts:
                        self._ref_targets.append((item.name, sub_item))

        total = len(self._ref_targets)
        for idx, (student_name, img_path) in enumerate(self._ref_targets, 1):
            if progress_callback:
                progress_callback(idx, total, img_path.name)

            image_data = self._load_image(img_path)
            if image_data is None:
                continue

            encs = self._get_face_encodings(image_data)
            if encs:
                if student_name not in self._student_encodings:
                    self._student_names.append(student_name)
                    self._student_encodings[student_name] = []
                self._student_encodings[student_name].extend(encs)
                self.logger.info(f"Loaded reference face for student: {student_name} ({img_path.name})")
            else:
                self.logger.warning(f"No face detected in reference photo: {img_path.name}")

    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        """Safely load image file into NumPy array."""
        try:
            if face_recognition is not None:
                return face_recognition.load_image_file(str(file_path))
            import PIL.Image
            img = PIL.Image.open(file_path).convert("RGB")
            return np.array(img)
        except Exception as e:
            self.logger.error(f"Failed to load image {file_path}: {e}")
            return None

    def _get_face_encodings(self, image_data: np.ndarray, num_jitters: int = 1) -> list[np.ndarray]:
        """Extract face encodings using explicit backend or face_recognition fallback."""
        if self.ai_backend in {"nvidia", "ollama"} and self.backend_mgr is not None:
            try:
                return self.backend_mgr.extract_encodings(image_data)
            except Exception as e:
                self.logger.warning(f"Backend '{self.ai_backend}' encoding failed, falling back: {e}")

        if face_recognition is not None:
            try:
                locations = face_recognition.face_locations(image_data)
                return face_recognition.face_encodings(image_data, locations, num_jitters=num_jitters)
            except Exception as e:
                self.logger.warning(f"face_recognition encoding failed: {e}")

        try:
            return self.backend_mgr.extract_encodings(image_data)
        except Exception as e:
            self.logger.error(f"Backend encoding extraction failed: {e}")
            return []

    def _compute_face_distances(self, reference_encodings: list[np.ndarray], target_encoding: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance safely using face_recognition or pure NumPy fallback."""
        if face_recognition is not None:
            return face_recognition.face_distance(reference_encodings, target_encoding)

        ref_arr = np.array(reference_encodings, dtype=np.float32)
        return np.linalg.norm(ref_arr - target_encoding, axis=1)

    def sort_all(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> dict[str, int]:
        summary = {"total": 0, "matched": 0, "unmatched": 0, "skipped": 0}

        image_files = [
            f for f in self.events_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        total_images = len(image_files)
        self.logger.info(f"Found {total_images} event photos to process")

        for idx, image_file in enumerate(image_files, 1):
            if cancelled and cancelled():
                self.logger.info("Sorting cancelled by user")
                break

            if progress_callback:
                progress_callback(idx, total_images, image_file.name)

            image_data = self._load_image(image_file)
            if image_data is None:
                summary["skipped"] += 1
                continue

            summary["total"] += 1
            encodings = self._get_face_encodings(image_data, num_jitters=self.num_jitters_match)

            if not encodings:
                self._copy_to_unmatched(image_file, "no_face_detected")
                summary["unmatched"] += 1
                if idx % self.gc_interval == 0:
                    gc.collect()
                continue

            matched_students = set()
            for face_encoding in encodings:
                for student_name in self._student_names:
                    student_encodings = self._student_encodings[student_name]
                    distances = self._compute_face_distances(student_encodings, face_encoding)
                    min_distance = float(np.min(distances))

                    if min_distance <= self.DISTANCE_THRESHOLD:
                        matched_students.add(student_name)

            if matched_students:
                for student_name in matched_students:
                    self._copy_to_student_folder(image_file, student_name)
                summary["matched"] += 1
            else:
                self._copy_to_unmatched(image_file, "no_match")
                summary["unmatched"] += 1

            del image_data, encodings
            if idx % self.gc_interval == 0:
                gc.collect()

        return summary

    def _generate_unique_filepath(self, target_folder: Path, prefix: str, original_name: str) -> Path:
        """Generate non-colliding output path."""
        stem = Path(original_name).stem
        ext = Path(original_name).suffix
        candidate = target_folder / f"{prefix}__{stem}{ext}"
        counter = 1
        while candidate.exists():
            candidate = target_folder / f"{prefix}__{stem}_{counter}{ext}"
            counter += 1
        return candidate

    def _copy_to_student_folder(self, image_path: Path, student_name: str) -> None:
        student_folder = self.output_dir / student_name
        student_folder.mkdir(parents=True, exist_ok=True)
        event_prefix = image_path.parent.name
        dest_path = self._generate_unique_filepath(student_folder, event_prefix, image_path.name)

        try:
            shutil.copy2(image_path, dest_path)
        except Exception as e:
            self.logger.error(f"Failed to copy {image_path.name} to {student_name}: {e}")

    def _copy_to_unmatched(self, image_path: Path, reason: str = "unmatched") -> None:
        unmatched_folder = self.output_dir / "_unmatched"
        unmatched_folder.mkdir(parents=True, exist_ok=True)
        dest_path = self._generate_unique_filepath(unmatched_folder, "unmatched", image_path.name)

        try:
            shutil.copy2(image_path, dest_path)
        except Exception as e:
            self.logger.error(f"Failed to copy {image_path.name} to unmatched: {e}")
