import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order four points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _warp_card(img: np.ndarray, quad: np.ndarray, W: int = 200, H: int = 300) -> np.ndarray:
    """Perspective-warp a quadrilateral region to a standard card size."""
    rect = _order_quad(quad.astype(np.float32))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    return warped

def detect_cards(bgr: np.ndarray, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Simple contour-based card detector.
    Returns list of dicts with keys: 'bbox', 'quad', 'warped'.
    - Assumes mostly top-down-ish shots with good lighting.
    """
    img = bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Canny edges
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    results = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 0.001 * (H * W):
            quad = approx.reshape(-1, 2).astype(np.float32)
            warped = _warp_card(img, quad)
            x, y, w, h = cv2.boundingRect(approx)
            results.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "quad": quad,
                "warped": warped
            })
    # Heuristic: sort left-to-right for deterministic order
    results.sort(key=lambda r: r["bbox"][0])
    return results

def extract_rank_suit_rois(warped: np.ndarray, corner_size: Tuple[int,int]=(70,100)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a warped card (200x300 default), crop the top-left corner that typically
    contains the rank & suit. Returns (rank_roi, suit_roi) as grayscale 64x64.
    """
    H, W = warped.shape[:2]
    cw, ch = corner_size
    corner = warped[0:ch, 0:cw]
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    # Basic normalization
    gray = cv2.resize(gray, (80, 120), interpolation=cv2.INTER_AREA)
    # Split: top portion for rank, lower for suit (very rough heuristic)
    rank = gray[0:70, 0:80]
    suit = gray[70:120, 0:80]
    rank = cv2.resize(rank, (64,64), interpolation=cv2.INTER_AREA)
    suit = cv2.resize(suit, (64,64), interpolation=cv2.INTER_AREA)
    return rank, suit
