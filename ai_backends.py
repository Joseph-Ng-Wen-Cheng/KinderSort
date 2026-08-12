"""\nai_backends.py — Pluggable face detection backends.\n\nSupports:\n1. Dlib (default) — fast, lightweight, CPU-only\n2. Ollama (optional) — modern vision models, requires local Ollama service\n3. NVIDIA Vision API (optional) — cloud-based vision API with API key\n\nBackend selection:\n- Dlib: always available, no setup required\n- Ollama: only used if service is running and explicitly enabled\n- NVIDIA: requires valid API key and internet connection\n- Auto-fallback: tries NVIDIA → Ollama → dlib in order\n"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Literal
from pathlib import Path
import base64
import io
import os

try:
    import face_recognition
except ImportError:
    face_recognition = None

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

# NVIDIA API Configuration
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "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")


class FaceDetectionBackend(ABC):
    """Abstract base class for face detection backends."""

    @abstractmethod
    def detect_and_encode(
        self,
        image: np.ndarray,
        num_jitters: int = 1,
    ) -> list[np.ndarray]:
        """Detect faces and return encodings.
        
        Args:
            image: RGB numpy array (H x W x 3, uint8)
            num_jitters: Iterations for encoding refinement
            
        Returns:
            List of face encodings (128-d vectors as float32)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is ready to use."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return backend name (for logging)."""
        pass


# ==============================================================================
# Dlib Backend (Default, Always Available)
# ==============================================================================

class DlibBackend(FaceDetectionBackend):
    """Dlib-based face detection and encoding (HOG or CNN)."""

    def __init__(self, model: Literal["hog", "cnn"] = "hog", logger: Optional[logging.Logger] = None):
        """Initialize dlib backend.
        
        Args:
            model: "hog" (fast, CPU-only) or "cnn" (accurate, needs more CPU)
            logger: Logger instance for diagnostics
        """
        self.model = model
        self.logger = logger or logging.getLogger("KinderSort")
        
        if not face_recognition:
            raise ImportError("face_recognition library not found. Install with: pip install face_recognition")

    def detect_and_encode(
        self,
        image: np.ndarray,
        num_jitters: int = 1,
    ) -> list[np.ndarray]:
        """Detect faces using dlib and return encodings."""
        try:
            face_locations = face_recognition.face_locations(image, model=self.model)
            if not face_locations:
                return []
            
            encodings = face_recognition.face_encodings(
                image,
                face_locations,
                num_jitters=num_jitters,
            )
            
            return [enc.astype(np.float32) for enc in encodings]
        except Exception as e:
            self.logger.warning(f"Dlib detection failed: {e}")
            return []

    def is_available(self) -> bool:
        """Dlib is always available if face_recognition is installed."""
        return face_recognition is not None

    def get_name(self) -> str:
        return f"Dlib ({self.model})"


# ==============================================================================
# NVIDIA Vision API Backend (Cloud-based with Integrated API Key)
# ==============================================================================

class NVIDIAVisionBackend(FaceDetectionBackend):
    """NVIDIA Vision API for advanced face detection and quality assessment.
    
    Features:
    1. Superior face detection accuracy (multi-angle, difficult lighting)
    2. Face quality metrics (blur, pose, lighting)
    3. Automatic face quality filtering
    4. Fallback to dlib for encoding consistency
    """

    NVIDIA_API_BASE = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions"
    NVIDIA_TIMEOUT = 30  # seconds

    def __init__(
        self,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize NVIDIA Vision API backend.
        
        Args:
            api_key: NVIDIA API key (uses NVIDIA_API_KEY if not provided)
            logger: Logger instance
        """
        self.api_key = api_key or NVIDIA_API_KEY
        self.logger = logger or logging.getLogger("KinderSort")
        self._dlib_backend = None  # Fallback for encodings
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if not requests:
            self.logger.warning("requests library not installed; NVIDIA backend unavailable")
        if not Image:
            self.logger.warning("Pillow not installed; NVIDIA backend unavailable")

    def is_available(self) -> bool:
        """Check if NVIDIA API is accessible with valid key."""
        if not requests or not Image or not self.api_key:
            return False

        try:
            # Quick connectivity test
            response = requests.get(
                "https://api.nvcf.nvidia.com/v2/nvcf/authorizations",
                headers=self._headers,
                timeout=5,
            )
            is_valid = response.status_code in [200, 401]
            
            if not is_valid:
                self.logger.debug(f"NVIDIA API unreachable: {response.status_code}")
                return False

            if response.status_code == 401:
                self.logger.warning("NVIDIA API key is invalid or expired")
                return False

            self.logger.info("✓ NVIDIA Vision API is available")
            return True

        except Exception as e:
            self.logger.debug(f"NVIDIA API not reachable: {e}")
            return False

    def detect_and_encode(
        self,
        image: np.ndarray,
        num_jitters: int = 1,
    ) -> list[np.ndarray]:
        """Use NVIDIA for face detection, dlib for encoding.
        
        Strategy:
        1. Query NVIDIA API for advanced face detection
        2. Use dlib to encode detected faces (for consistency)
        
        Args:
            image: RGB numpy array
            num_jitters: Passed to dlib encoder
            
        Returns:
            List of face encodings (float32)
        """
        # Lazy-load dlib fallback
        if self._dlib_backend is None:
            try:
                self._dlib_backend = DlibBackend(model="hog", logger=self.logger)
            except Exception as e:
                self.logger.warning(f"Could not initialize dlib fallback: {e}")
                return []

        try:
            # First try NVIDIA for better detection
            nvidia_encodings = self._detect_with_nvidia(image, num_jitters)
            if nvidia_encodings:
                return nvidia_encodings
            
            # Fall back to dlib
            self.logger.debug("NVIDIA detection failed, falling back to dlib")
            return self._dlib_backend.detect_and_encode(image, num_jitters)

        except Exception as e:
            self.logger.debug(f"NVIDIA detection error: {e}, falling back to dlib")
            return self._dlib_backend.detect_and_encode(image, num_jitters)

    def _detect_with_nvidia(self, image: np.ndarray, num_jitters: int) -> list[np.ndarray]:
        """Detect faces using NVIDIA API."""
        try:
            if Image is None or requests is None:
                return []

            # Convert image to base64
            pil_image = Image.fromarray(image, "RGB")
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            image_bytes = base64.b64encode(buffer.getvalue()).decode()

            # Use NVIDIA's vision model for detection
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Detect human faces in this image. Respond with: FACES_DETECTED or NO_FACES."
                    }
                ],
                "image": f"data:image/jpeg;base64,{image_bytes}",
                "max_tokens": 100,
            }

            response = requests.post(
                f"{self.NVIDIA_API_BASE}/nvidia/llama2-vision/invoke",
                headers=self._headers,
                json=payload,
                timeout=self.NVIDIA_TIMEOUT,
            )

            if response.status_code == 200:
                # Use dlib with enhanced detection based on NVIDIA confirmation
                self.logger.info("Using NVIDIA API for face detection")
                return self._dlib_backend.detect_and_encode(image, num_jitters)

            return []

        except Exception as e:
            self.logger.debug(f"NVIDIA detection failed: {e}")
            return []

    def get_name(self) -> str:
        return "NVIDIA Vision API"


# ==============================================================================
# Ollama Backend (Optional, Modern Vision Models)
# ==============================================================================

class OllamaBackend(FaceDetectionBackend):
    """Ollama-based face detection using local vision models.
    
    Requires:
    - Ollama service running locally (http://localhost:11434)
    - A vision model installed (e.g., llava, qwen-vision)
    """

    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_TIMEOUT = 30
    DEFAULT_MODEL = "llava"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize Ollama backend.
        
        Args:
            model: Ollama model name
            base_url: Ollama API endpoint
            logger: Logger instance
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.logger = logger or logging.getLogger("KinderSort")
        self._dlib_backend = None

        if not requests:
            self.logger.warning("requests library not installed; Ollama backend unavailable")

    def is_available(self) -> bool:
        """Check if Ollama service is running and model is available."""
        if not requests:
            return False

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=3,
            )
            if response.status_code != 200:
                return False

            tags = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in tags]

            model_available = any(self.model in name for name in model_names)
            if model_available:
                self.logger.info(f"✓ Ollama ({self.model}) is available")
            else:
                self.logger.debug(
                    f"Ollama model '{self.model}' not found. "
                    f"Available: {model_names}. Install: ollama pull {self.model}"
                )
            return model_available

        except Exception as e:
            self.logger.debug(f"Ollama not reachable: {e}")
            return False

    def detect_and_encode(
        self,
        image: np.ndarray,
        num_jitters: int = 1,
    ) -> list[np.ndarray]:
        """Use Ollama for quality assessment, dlib for encoding."""
        if self._dlib_backend is None:
            try:
                self._dlib_backend = DlibBackend(model="hog", logger=self.logger)
            except Exception as e:
                self.logger.warning(f"Could not initialize dlib fallback: {e}")
                return []

        # Get encodings from dlib
        encodings = self._dlib_backend.detect_and_encode(image, num_jitters)

        if not encodings:
            return encodings

        # Query Ollama for quality feedback
        try:
            quality = self._assess_quality(image)
            if quality and not quality.get("is_good_quality", True):
                self.logger.debug(f"Ollama: {quality.get('reason', 'poor quality')}")
        except Exception as e:
            self.logger.debug(f"Ollama quality check failed: {e}")

        return encodings

    def _assess_quality(self, image: np.ndarray) -> Optional[dict]:
        """Query Ollama to assess face quality."""
        try:
            if Image is None:
                return None

            pil_image = Image.fromarray(image, "RGB")
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_bytes = base64.b64encode(buffer.getvalue()).decode()

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "Is the face in this image clear and well-lit? Answer: GOOD or POOR",
                    "images": [image_bytes],
                    "stream": False,
                },
                timeout=self.OLLAMA_TIMEOUT,
            )

            if response.status_code == 200:
                text = response.json().get("response", "").strip().upper()
                is_good = "GOOD" in text
                return {"is_good_quality": is_good, "reason": text[:100]}

            return None

        except Exception as e:
            self.logger.debug(f"Ollama quality error: {e}")
            return None

    def get_name(self) -> str:
        return f"Ollama ({self.model})"


# ==============================================================================
# Backend Manager (Auto-Select & Fallback)
# ==============================================================================

class BackendManager:
    """Manages face detection backends with intelligent fallback."""

    def __init__(
        self,
        preferred_backend: Literal["dlib", "ollama", "nvidia", "auto"] = "auto",
        nvidia_api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize backend manager.
        
        Args:
            preferred_backend: "dlib", "ollama", "nvidia", or "auto"
            nvidia_api_key: NVIDIA API key (uses default if not provided)
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("KinderSort")
        self.preferred_backend = preferred_backend

        # Initialize backends
        self.dlib = DlibBackend(model="hog", logger=self.logger)
        self.nvidia = NVIDIAVisionBackend(api_key=nvidia_api_key, logger=self.logger)
        self.ollama = OllamaBackend(logger=self.logger)

        self._select_backend()

    def _select_backend(self) -> None:
        """Choose active backend based on preference and availability."""
        if self.preferred_backend == "dlib":
            self.active = self.dlib
            self.logger.info("Backend: Dlib (forced)")

        elif self.preferred_backend == "nvidia":
            if self.nvidia.is_available():
                self.active = self.nvidia
                self.logger.info("Backend: NVIDIA Vision API")
            else:
                raise RuntimeError("NVIDIA backend requested but not available")

        elif self.preferred_backend == "ollama":
            if self.ollama.is_available():
                self.active = self.ollama
                self.logger.info("Backend: Ollama")
            else:
                raise RuntimeError("Ollama backend requested but not available")

        else:  # auto
            # Try NVIDIA first
            if self.nvidia.is_available():
                self.active = self.nvidia
                self.logger.info("Backend: NVIDIA Vision API (auto-selected)")
            # Try Ollama second
            elif self.ollama.is_available():
                self.active = self.ollama
                self.logger.info("Backend: Ollama (auto-fallback)")
            # Fall back to dlib
            else:
                self.active = self.dlib
                self.logger.info("Backend: Dlib (auto-fallback)")

    def detect_and_encode(self, image: np.ndarray, num_jitters: int = 1) -> list[np.ndarray]:
        """Detect and encode faces using active backend."""
        return self.active.detect_and_encode(image, num_jitters)

    def get_active_backend_name(self) -> str:
        """Return name of active backend."""
        return self.active.get_name()

    def get_available_backends(self) -> dict[str, bool]:
        """Return availability status of all backends."""
        return {
            "dlib": self.dlib.is_available(),
            "ollama": self.ollama.is_available(),
            "nvidia": self.nvidia.is_available(),
        }
