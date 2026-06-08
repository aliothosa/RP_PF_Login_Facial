"""Pruebas de FaceDetector.

``select_largest_face`` se prueba con cajas sintéticas (sin cámara ni cascade).
La construcción del detector y ``detect_faces`` usan OpenCV; se omiten si cv2 no
está disponible en el entorno.
"""

import pytest

from rp_face_login.vision.face_detector import FaceDetector


# --- select_largest_face (cajas sintéticas, sin dependencias de hardware) ---

def test_select_largest_face_picks_max_area():
    boxes = [
        (0, 0, 10, 10),     # área 100
        (5, 5, 50, 40),     # área 2000  <- mayor
        (1, 1, 30, 30),     # área 900
    ]
    assert FaceDetector.select_largest_face(boxes) == (5, 5, 50, 40)


def test_select_largest_face_area_is_width_times_height():
    # Una caja ancha y baja (área 1200) debe ganar a una alta y angosta (área 1000).
    boxes = [(0, 0, 10, 100), (0, 0, 60, 20)]
    assert FaceDetector.select_largest_face(boxes) == (0, 0, 60, 20)


def test_select_largest_face_single_box():
    assert FaceDetector.select_largest_face([(2, 3, 7, 8)]) == (2, 3, 7, 8)


def test_select_largest_face_empty_returns_none():
    assert FaceDetector.select_largest_face([]) is None


def test_select_largest_face_none_returns_none():
    assert FaceDetector.select_largest_face(None) is None


def test_select_largest_face_tie_returns_a_valid_box():
    boxes = [(0, 0, 10, 10), (5, 5, 10, 10)]  # mismo área
    assert FaceDetector.select_largest_face(boxes) in boxes


# --- Detector real (requiere OpenCV) ---

def test_detector_constructs_and_detects_nothing_on_blank_image():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    detector = FaceDetector()
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes = detector.detect_faces(blank)
    assert boxes == []
    assert detector.select_largest_face(boxes) is None


def test_detect_faces_rejects_none_frame():
    pytest.importorskip("cv2")
    detector = FaceDetector()
    with pytest.raises(ValueError):
        detector.detect_faces(None)
