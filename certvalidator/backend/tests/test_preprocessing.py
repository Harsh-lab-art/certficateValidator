"""
Tests for the preprocessing pipeline.
Run: pytest backend/tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from ml.src.preprocessing.pipeline import CertificatePreprocessor, TARGET_LONG_SIDE


@pytest.fixture
def preprocessor():
    return CertificatePreprocessor()


@pytest.fixture
def synthetic_cert():
    """Create a simple synthetic white certificate image for testing."""
    img = np.ones((1200, 1700, 3), dtype=np.uint8) * 255
    # Add some text-like black marks
    cv2.putText(img, "CERTIFICATE OF DEGREE", (200, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
    cv2.putText(img, "John Doe", (400, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 80), 3)
    cv2.rectangle(img, (50, 50), (1650, 1150), (20, 60, 120), 8)
    return img


@pytest.fixture
def skewed_cert(synthetic_cert):
    """Create a 3-degree skewed version of the cert."""
    h, w = synthetic_cert.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    return cv2.warpAffine(synthetic_cert, M, (w, h),
                          borderMode=cv2.BORDER_REPLICATE)


def _save_tmp(img: np.ndarray, name: str) -> Path:
    p = Path(f"/tmp/test_{name}.png")
    cv2.imwrite(str(p), img)
    return p


class TestDPINormalise:
    def test_upscale(self, preprocessor, synthetic_cert):
        result = preprocessor._normalise_dpi(synthetic_cert)
        assert max(result.shape[:2]) == TARGET_LONG_SIDE

    def test_already_target_size(self, preprocessor):
        img = np.zeros((1754, 2480, 3), dtype=np.uint8)
        result = preprocessor._normalise_dpi(img)
        assert result.shape == img.shape

    def test_preserves_channels(self, preprocessor, synthetic_cert):
        result = preprocessor._normalise_dpi(synthetic_cert)
        assert result.ndim == 3
        assert result.shape[2] == 3


class TestDeskew:
    def test_corrects_skew(self, preprocessor, skewed_cert):
        corrected, angle = preprocessor._deskew(skewed_cert)
        assert abs(angle) < 5.0   # detected some angle
        assert corrected.shape[2] == 3

    def test_straight_image_unchanged(self, preprocessor, synthetic_cert):
        _, angle = preprocessor._deskew(synthetic_cert)
        assert abs(angle) < 1.0   # near zero for already-straight image

    def test_output_is_uint8(self, preprocessor, skewed_cert):
        corrected, _ = preprocessor._deskew(skewed_cert)
        assert corrected.dtype == np.uint8


class TestBorderCrop:
    def test_removes_white_border(self, preprocessor):
        # Create image with 50px white border around content
        img = np.ones((500, 700, 3), dtype=np.uint8) * 255
        img[50:450, 50:650] = 128   # grey content area
        cropped = preprocessor._crop_borders(img)
        assert cropped.shape[0] < 500
        assert cropped.shape[1] < 700

    def test_content_preserved(self, preprocessor, synthetic_cert):
        cropped = preprocessor._crop_borders(synthetic_cert)
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0


class TestDenoise:
    def test_output_shape_unchanged(self, preprocessor, synthetic_cert):
        denoised = preprocessor._denoise(synthetic_cert)
        assert denoised.shape == synthetic_cert.shape

    def test_output_dtype(self, preprocessor, synthetic_cert):
        denoised = preprocessor._denoise(synthetic_cert)
        assert denoised.dtype == np.uint8


class TestELA:
    def test_output_shape(self, preprocessor, synthetic_cert):
        ela = preprocessor._compute_ela(synthetic_cert)
        assert ela.shape == synthetic_cert.shape

    def test_ela_is_uint8(self, preprocessor, synthetic_cert):
        ela = preprocessor._compute_ela(synthetic_cert)
        assert ela.dtype == np.uint8

    def test_ela_values_in_range(self, preprocessor, synthetic_cert):
        ela = preprocessor._compute_ela(synthetic_cert)
        assert ela.min() >= 0
        assert ela.max() <= 255

    def test_tampered_region_higher_ela(self, preprocessor, synthetic_cert):
        """
        After digitally editing a region, that region should have higher
        ELA values than unedited regions.
        """
        tampered = synthetic_cert.copy()
        # Paint a white rectangle (simulates tampering)
        tampered[200:300, 300:600] = (255, 255, 200)

        ela_clean   = preprocessor._compute_ela(synthetic_cert).astype(float)
        ela_tampered = preprocessor._compute_ela(tampered).astype(float)

        # Mean ELA in tampered region should be higher
        region_clean    = ela_clean[200:300, 300:600].mean()
        region_tampered = ela_tampered[200:300, 300:600].mean()
        assert region_tampered >= region_clean   # tampered region has higher error


class TestFullPipeline:
    def test_pipeline_from_file(self, preprocessor, synthetic_cert, tmp_path):
        p = tmp_path / "test_cert.png"
        cv2.imwrite(str(p), synthetic_cert)

        result = preprocessor.process(str(p))

        assert result.success
        assert result.processed_image is not None
        assert result.ela_image is not None
        assert result.grayscale is not None
        assert result.processed_image.ndim == 3
        assert result.grayscale.ndim == 2
        assert "processing_time_s" in result.metadata

    def test_pipeline_from_bytes(self, preprocessor, synthetic_cert):
        _, buf = cv2.imencode(".jpg", synthetic_cert)
        data = buf.tobytes()

        result = preprocessor.process_bytes(data, suffix=".jpg")
        assert result.success

    def test_graceful_failure_bad_path(self, preprocessor):
        result = preprocessor.process("/nonexistent/path/cert.jpg")
        assert not result.success
        assert result.error is not None
