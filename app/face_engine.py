"""Face detection + recognition engine for FRS SmartScore.

Uses InsightFace `buffalo_l` (SCRFD detector + ArcFace R50 recognizer) on ONNX Runtime.
NOTE: the pretrained buffalo_l weights are licensed for non-commercial research use only.
Obtain a commercial licence (or swap the model) before production use.
"""

from __future__ import annotations

import base64
import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

MODEL_NAME = os.getenv("FRS_MODEL", "buffalo_l")
MODEL_ROOT = os.getenv("FRS_MODEL_ROOT", os.path.join(os.path.dirname(__file__), "..", "models"))

# Cosine-similarity thresholds (tune on your own selfie-vs-ID pairs).
MATCH_THRESHOLD = float(os.getenv("FRS_MATCH_THRESHOLD", "0.35"))
REVIEW_THRESHOLD = float(os.getenv("FRS_REVIEW_THRESHOLD", "0.25"))

MIN_FACE_PX = 100  # faces smaller than this are upscaled before recognition
MAX_SIDE_PX = 2000  # very large photos are downscaled first (speed)
UPRIGHT_ACCEPT_CONF = 0.85  # face at 0 deg this confident -> skip rotation search
ROTATION_ZERO_BONUS = 0.05  # 0 deg wins unless a rotation is clearly more confident


class FaceNotFoundError(Exception):
    pass


@dataclass
class FaceResult:
    embedding: np.ndarray
    bbox: list[int]
    det_score: float
    face_px: int
    faces_found: int
    rotation: int
    warnings: list[str] = field(default_factory=list)
    thumbnail_b64: str = ""

    def public(self) -> dict:
        return {
            "faces_found": self.faces_found,
            "face_box": self.bbox,
            "face_size_px": self.face_px,
            "detection_confidence": round(self.det_score, 3),
            "rotation_applied": self.rotation,
            "warnings": self.warnings,
            "face_thumbnail": self.thumbnail_b64,
        }


class FaceEngine:
    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=MODEL_NAME,
            root=MODEL_ROOT,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.4)
        self._lock = threading.Lock()

    # ---------- public API ----------

    def analyze(self, img: np.ndarray) -> FaceResult:
        img = _limit_size(img)
        faces, used, rotation = self._detect_with_fallbacks(img)
        if not faces:
            raise FaceNotFoundError("No face detected")

        face = _pick_primary(faces)
        warnings: list[str] = []
        if len(faces) > 1:
            warnings.append(f"{len(faces)} faces found - using the largest one")

        x1, y1, x2, y2 = face.bbox.astype(int)
        face_px = int(min(x2 - x1, y2 - y1))

        # Small faces (typical on ID cards): upscale and re-detect for better alignment.
        if face_px < MIN_FACE_PX:
            refined = self._refine_small_face(used, face)
            if refined is not None:
                face = refined
            warnings.append(f"Face is small ({face_px}px) - image upscaled; accuracy may be lower")

        if face.det_score < 0.6:
            warnings.append("Low face-detection confidence")

        crop = _crop(used, face.bbox)
        blur = _blur_score(crop)
        if blur < 40:
            warnings.append("Face looks blurry")
        brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()) if crop.size else 0
        if brightness < 50:
            warnings.append("Face is too dark")
        elif brightness > 215:
            warnings.append("Face is over-exposed")

        return FaceResult(
            embedding=face.normed_embedding,
            bbox=[int(v) for v in face.bbox],
            det_score=float(face.det_score),
            face_px=face_px,
            faces_found=len(faces),
            rotation=rotation,
            warnings=warnings,
            thumbnail_b64=_thumb_b64(crop),
        )

    @staticmethod
    def compare(a: FaceResult, b: FaceResult) -> dict:
        similarity = float(np.clip(np.dot(a.embedding, b.embedding), -1.0, 1.0))
        score = similarity_to_score(similarity)
        if similarity >= MATCH_THRESHOLD:
            decision = "MATCH"
        elif similarity >= REVIEW_THRESHOLD:
            decision = "REVIEW"
        else:
            decision = "NO_MATCH"
        return {
            "frs_score": score,
            "decision": decision,
            "cosine_similarity": round(similarity, 4),
            "thresholds": {"match": MATCH_THRESHOLD, "review": REVIEW_THRESHOLD},
        }

    # ---------- internals ----------

    def _get(self, img: np.ndarray):
        with self._lock:  # FaceAnalysis keeps per-call state; keep it simple and safe
            return self._app.get(img)

    def _detect_with_fallbacks(self, img: np.ndarray):
        """Find the correctly oriented face (ID cards are often photographed sideways).

        The detector also fires on sideways/upside-down faces, and its landmarks then
        still *look* upright, so we cannot trust the first hit. Instead: if the face at
        0 deg is clearly confident, accept it; otherwise try all 4 rotations and keep the
        most confident one (0 deg gets a small bonus so normal photos are never flipped).
        """
        candidates = []
        for rotation in (0, 90, 270, 180):
            rotated = _rotate(img, rotation)
            faces, used = self._get(rotated), rotated
            if not faces and max(rotated.shape[:2]) < 1200:  # tiny face may need upscaling
                used = cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                faces = self._get(used)
            if not faces:
                continue
            conf = float(_pick_primary(faces).det_score)
            if rotation == 0:
                if conf >= UPRIGHT_ACCEPT_CONF:
                    return faces, used, 0
                conf += ROTATION_ZERO_BONUS
            candidates.append((conf, faces, used, rotation))
        if candidates:
            _, faces, used, rotation = max(candidates, key=lambda c: c[0])
            return faces, used, rotation
        # Tightly cropped face (no background around it): pad, enlarge and retry.
        h, w = img.shape[:2]
        p = max(h, w) // 2
        padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if max(padded.shape[:2]) < 640:
            s = 640 / max(padded.shape[:2])
            padded = cv2.resize(padded, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        faces = self._get(padded)
        if faces:
            return faces, padded, 0
        return [], img, 0

    def _refine_small_face(self, img: np.ndarray, face):
        x1, y1, x2, y2 = face.bbox
        w, h = x2 - x1, y2 - y1
        pad = 0.8
        H, W = img.shape[:2]
        cx1, cy1 = int(max(0, x1 - w * pad)), int(max(0, y1 - h * pad))
        cx2, cy2 = int(min(W, x2 + w * pad)), int(min(H, y2 + h * pad))
        region = img[cy1:cy2, cx1:cx2]
        if region.size == 0:
            return None
        scale = max(1.0, 220.0 / max(1.0, min(w, h)))
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        region = _sharpen(region)
        faces = self._get(region)
        if not faces:
            return None
        best = _pick_primary(faces)
        # map bbox back to original coordinates (embedding is already computed on the upscaled crop)
        best.bbox = np.array(best.bbox) / scale + np.array([cx1, cy1, cx1, cy1])
        return best


# ---------- helpers ----------

def similarity_to_score(sim: float) -> float:
    """Map ArcFace cosine similarity to a 0-100 score.

    Anchors: review threshold -> 50, match threshold -> 70, 0.60 -> 95, 0.80 -> 100.
    """
    xs = [0.0, REVIEW_THRESHOLD, MATCH_THRESHOLD, 0.60, 0.80]
    ys = [0.0, 50.0, 70.0, 95.0, 100.0]
    return round(float(np.interp(sim, xs, ys)), 1)


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # honours EXIF orientation
    if img is None:
        raise ValueError("Could not read image (unsupported or corrupt file)")
    return img


def _pick_primary(faces):
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)

    return max(faces, key=lambda f: area(f) * f.det_score)


def _limit_size(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= MAX_SIDE_PX:
        return img
    s = MAX_SIDE_PX / max(h, w)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _rotate(img: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _sharpen(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def _crop(img: np.ndarray, bbox) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    m = int(0.15 * max(x2 - x1, y2 - y1))
    return img[max(0, y1 - m):min(H, y2 + m), max(0, x1 - m):min(W, x2 + m)]


def _blur_score(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _thumb_b64(crop: np.ndarray) -> str:
    if crop.size == 0:
        return ""
    t = cv2.resize(crop, (160, int(160 * crop.shape[0] / max(1, crop.shape[1]))))
    ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""
