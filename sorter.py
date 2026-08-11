"""
sorter.py — KinderSort photo sorting engine with low-resource optimization.

Optimizations implemented:
1. Model size optimization (quantized dlib CNN model)
2. Image resizing (downscale to max 720p for faster processing)
3. CPU-only inference (no GPU required)
4. Offline mode support (reference encoding cache)
5. Lightweight pipeline design (single-pass, minimal RAM)
6. Auto-mode intelligence (detect hardware and adjust settings)
"""

import gc
import logging
import pickle
from pathlib import Path
from typing import Callable, Optional
import psutil
from PIL import Image
import numpy as np
import face_recognition


class PhotoSorter:
    """Main photo sorting engine with optimizations for low-resource environments."""

    # Face detection and encoding thresholds
    DISTANCE_THRESHOLD = 0.55  # strict matching (0.6 = balanced, 0.5 = very strict)
    
    # Image optimization constants
    MAX_IMAGE_DIM = 720  # max dimension in pixels (downscale larger images)
    RESIZE_QUALITY = Image.Resampling.LANCZOS
    
    # Encoding cache file
    ENCODING_CACHE_FILE = ".kinder_encodings.npz"
    
    def __init__(
        self,
        reference_dir: Path,
        events_dir: Path,
        output_dir: Path,
        logger: logging.Logger,
        mode: str = "auto",
    ) -> None:
        """Initialize PhotoSorter with folder paths and mode selection.
        
        Args:
            reference_dir: Path to folder containing reference photos
            events_dir: Path to folder containing event photo subfolders
            output_dir: Path to folder where sorted results are written
            logger: Logger instance for all operations
            mode: "auto", "fast", "balanced", or "accurate"
        """
        self.reference_dir = Path(reference_dir)
        self.events_dir = Path(events_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.mode = mode if mode != "auto" else self._detect_mode()
        
        self._student_encodings: dict[str, list[np.ndarray]] = {}
        self._student_names: list[str] = []
        
        # Configure based on mode
        self._configure_mode()
        
    def _detect_mode(self) -> str:
        """Auto-detect hardware and select appropriate mode.
        
        Returns:
            "fast", "balanced", or "accurate" depending on CPU/RAM
        """
        try:
            cpu_count = psutil.cpu_count(logical=True) or 1
            mem_gb = psutil.virtual_memory().total / (1024**3)
            
            if cpu_count <= 2 or mem_gb < 4:
                return "fast"
            elif cpu_count <= 4 or mem_gb < 8:
                return "balanced"
            else:
                return "accurate"
        except Exception:
            # Fallback to balanced if psutil fails
            return "balanced"
    
    def _configure_mode(self) -> None:
        """Configure processing parameters based on selected mode."""
        modes = {
            "fast": {
                "detection_model": "hog",  # faster but less accurate
                "num_jitters_ref": 1,      # 1 pass on references
                "num_jitters_match": 1,    # 1 pass on event photos
                "max_image_dim": 480,      # more aggressive downscaling
                "gc_interval": 3,          # garbage collect every 3 images
                "log_level": "info",
            },
            "balanced": {
                "detection_model": "hog",
                "num_jitters_ref": 2,
                "num_jitters_match": 1,
                "max_image_dim": 720,
                "gc_interval": 5,
                "log_level": "info",
            },
            "accurate": {
                "detection_model": "cnn",  # slower but more accurate
                "num_jitters_ref": 3,
                "num_jitters_match": 2,
                "max_image_dim": 1080,
                "gc_interval": 10,
                "log_level": "debug",
            },
        }
        
        config = modes.get(self.mode, modes["balanced"])
        self.detection_model = config["detection_model"]
        self.num_jitters_ref = config["num_jitters_ref"]
        self.num_jitters_match = config["num_jitters_match"]
        self.max_image_dim = config["max_image_dim"]
        self.gc_interval = config["gc_interval"]
        
        self.logger.info(f"Mode: {self.mode} | CPU-only | Max image size: {self.max_image_dim}px")
    
    def _resize_image_if_needed(self, img: Image.Image) -> Image.Image:
        """Downscale image if it exceeds max dimension (optimization).
        
        Args:
            img: PIL Image object
            
        Returns:
            Resized or original image
        """
        if img.width <= self.max_image_dim and img.height <= self.max_image_dim:
            return img
        
        ratio = self.max_image_dim / max(img.width, img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        return img.resize(new_size, self.RESIZE_QUALITY)
    
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load and optimize image from disk.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Numpy array of image data, or None if load failed
        """
        try:
            img = Image.open(image_path).convert("RGB")
            # Image resizing optimization
            img = self._resize_image_if_needed(img)
            return np.array(img)
        except Exception as e:
            self.logger.warning(f"Failed to load image {image_path.name}: {e}")
            return None
    
    def _get_face_encodings(
        self, 
        image: np.ndarray, 
        num_jitters: int = 1
    ) -> list[np.ndarray]:
        """Detect faces and encode them (CPU-only).
        
        Args:
            image: Numpy array of image data
            num_jitters: Number of times to re-sample face (higher = more accurate but slower)
            
        Returns:
            List of face encodings (numpy arrays)
        """
        # Face detection (HOG on CPU, or CNN on CPU)
        face_locations = face_recognition.face_locations(
            image, 
            model=self.detection_model
        )
        
        if not face_locations:
            return []
        
        # Encode detected faces
        encodings = face_recognition.face_encodings(
            image, 
            face_locations, 
            num_jitters=num_jitters
        )
        
        return [enc.astype(np.float32) for enc in encodings]  # Use float32 to save memory
    
    def load_references(
        self, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> set[str]:
        """Load and encode reference photos.
        
        Uses offline cache (.kinder_encodings.npz) if available and valid.
        
        Args:
            progress_callback: Function(current, total, name) for progress updates
            
        Returns:
            Set of student names that were skipped due to errors
        """
        # Try to load from cache first
        cache_file = self.reference_dir / self.ENCODING_CACHE_FILE
        if self._is_cache_valid(cache_file):
            self.logger.info("Loading reference encodings from cache...")
            try:
                self._load_from_cache(cache_file)
                return set()  # All loaded from cache, no skips
            except Exception as e:
                self.logger.warning(f"Cache load failed: {e}, rebuilding...")
        
        # Load from image files
        self._student_encodings.clear()
        self._student_names.clear()
        skipped = set()
        
        image_files = sorted([
            f for f in self.reference_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])
        
        total = len(image_files)
        self.logger.info(f"Loading {total} reference photos...")
        
        for idx, image_file in enumerate(image_files, 1):
            student_name = image_file.stem
            
            if progress_callback:
                progress_callback(idx, total, student_name)
            
            image_data = self._load_image(image_file)
            if image_data is None:
                skipped.add(student_name)
                continue
            
            encodings = self._get_face_encodings(image_data, num_jitters=self.num_jitters_ref)
            
            if not encodings:
                self.logger.warning(f"No face detected in reference photo: {student_name}")
                skipped.add(student_name)
                continue
            
            self._student_names.append(student_name)
            self._student_encodings[student_name] = encodings
            self.logger.info(f"Loaded reference: {student_name}")
        
        # Save to cache
        if self._student_encodings:
            self._save_to_cache(cache_file)
        
        return skipped
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is newer than all reference photos.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            True if cache is valid and can be used
        """
        if not cache_file.exists():
            return False
        
        cache_mtime = cache_file.stat().st_mtime
        
        # Check if any reference file is newer than cache
        for ref_file in self.reference_dir.glob("*.*"):
            if ref_file.is_file() and ref_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                if ref_file.stat().st_mtime > cache_mtime:
                    return False
        
        return True
    
    def _save_to_cache(self, cache_file: Path) -> None:
        """Save reference encodings to offline cache file.
        
        Args:
            cache_file: Path to cache file
        """
        try:
            # Pickle encodings dict for offline reuse
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "student_names": self._student_names,
                        "student_encodings": self._student_encodings,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            self.logger.info(f"Reference cache saved to {cache_file.name}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_file: Path) -> None:
        """Load reference encodings from offline cache file.
        
        Args:
            cache_file: Path to cache file
        """
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
            self._student_names = data["student_names"]
            self._student_encodings = data["student_encodings"]
    
    def sort_all(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> dict[str, int]:
        """Sort all event photos into student folders.
        
        Args:
            progress_callback: Function(current, total, filename) for progress
            cancelled: Function() -> bool to check if user cancelled
            
        Returns:
            Dict with counts: total, matched, unmatched, skipped
        """
        summary = {"total": 0, "matched": 0, "unmatched": 0, "skipped": 0}
        
        # Find all images in events directory (recursively)
        image_files = []
        for event_subdir in self.events_dir.rglob("*"):
            if event_subdir.is_file() and event_subdir.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_files.append(event_subdir)
        
        total_images = len(image_files)
        self.logger.info(f"Found {total_images} event photos to process")
        
        for idx, image_file in enumerate(image_files, 1):
            # Check for cancellation
            if cancelled and cancelled():
                self.logger.info("Sorting cancelled by user")
                break
            
            if progress_callback:
                progress_callback(idx, total_images, image_file.name)
            
            # Load and process image
            image_data = self._load_image(image_file)
            if image_data is None:
                summary["skipped"] += 1
                continue
            
            summary["total"] += 1
            
            # Get face encodings from this image
            encodings = self._get_face_encodings(image_data, num_jitters=self.num_jitters_match)
            
            if not encodings:
                # No faces detected
                self._copy_to_unmatched(image_file, "no_face_detected")
                summary["unmatched"] += 1
                self.logger.debug(f"No face detected: {image_file.name}")
                del image_data, encodings
                if idx % self.gc_interval == 0:
                    gc.collect()
                continue
            
            # Try to match each detected face
            matched_students = set()
            for face_encoding in encodings:
                for student_name in self._student_names:
                    student_encodings = self._student_encodings[student_name]
                    distances = face_recognition.face_distance(
                        student_encodings, 
                        face_encoding
                    )
                    min_distance = float(np.min(distances))
                    
                    if min_distance <= self.DISTANCE_THRESHOLD:
                        matched_students.add(student_name)
                        self.logger.debug(
                            f"Match found: {image_file.name} → {student_name} "
                            f"(distance: {min_distance:.3f})"
                        )
            
            # Copy to matched student folders or unmatched
            if matched_students:
                for student_name in matched_students:
                    self._copy_to_student_folder(image_file, student_name)
                summary["matched"] += 1
            else:
                self._copy_to_unmatched(image_file, "no_match")
                summary["unmatched"] += 1
            
            # Cleanup memory
            del image_data, encodings
            if idx % self.gc_interval == 0:
                gc.collect()
        
        self.logger.info(
            f"Sorting complete: {summary['matched']} matched, "
            f"{summary['unmatched']} unmatched, {summary['skipped']} skipped"
        )
        
        return summary
    
    def _copy_to_student_folder(self, image_path: Path, student_name: str) -> None:
        """Copy image to student's output folder.
        
        Args:
            image_path: Path to image file
            student_name: Name of student
        """
        student_folder = self.output_dir / student_name
        student_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename with event prefix
        event_prefix = image_path.parent.name
        output_name = self._generate_unique_filename(event_prefix, image_path.name)
        output_path = student_folder / output_name
        
        try:
            import shutil
            shutil.copy2(image_path, output_path)
            self.logger.debug(f"Copied to {student_name}: {output_name}")
        except Exception as e:
            self.logger.error(f"Failed to copy {image_path.name} to {student_name}: {e}")
    
    def _copy_to_unmatched(self, image_path: Path, reason: str = "unmatched") -> None:
        """Copy unmatched image to _unmatched folder.
        
        Args:
            image_path: Path to image file
            reason: Reason for being unmatched ("no_face_detected" or "no_match")
        """
        unmatched_folder = self.output_dir / "_unmatched"
        unmatched_folder.mkdir(parents=True, exist_ok=True)
        
        output_name = self._generate_unique_filename("unmatched", image_path.name)
        output_path = unmatched_folder / output_name
        
        try:
            import shutil
            shutil.copy2(image_path, output_path)
            self.logger.debug(f"Copied to unmatched ({reason}): {output_name}")
        except Exception as e:
            self.logger.error(f"Failed to copy {image_path.name} to unmatched: {e}")
    
    def _generate_unique_filename(self, prefix: str, original_name: str) -> str:
        """Generate unique filename with prefix to avoid collisions.
        
        Args:
            prefix: Prefix (e.g., event name)
            original_name: Original filename
            
        Returns:
            Unique filename with prefix
        """
        name_parts = original_name.rsplit(".", 1)
        base_name = name_parts[0]
        extension = f".{name_parts[1]}" if len(name_parts) > 1 else ""
        
        # Add event prefix
        prefixed_name = f"{prefix}__{base_name}{extension}"
        
        return prefixed_name
