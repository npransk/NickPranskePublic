import argparse
import os
import cv2
import pytesseract
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

# If tesseract is not on PATH (Windows), uncomment and set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ALLOWED = set("FOX")

@dataclass
class CharBox:
    ch: str
    x1: int
    y1: int
    x2: int
    y2: int
    xc: int
    yc: int
    w: int
    h: int

@dataclass
class FoxHit:
    direction: str  # 'H', 'V', 'D1' (down-right), 'D2' (down-left)
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    letters: List[CharBox]

def load_and_binarize(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gentle denoise + contrast
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold handles uneven lighting well
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )

    # If text is light, invert so text is dark on white (tesseract works either way,
    # but boxes tend to be a bit cleaner with dark text)
    # Heuristic: if more than half pixels are black, invert
    if (bw < 128).mean() > 0.5:
        bw = cv2.bitwise_not(bw)

    return bw

def ocr_char_boxes(bw: np.ndarray) -> List[CharBox]:
    """
    Use Tesseract to get per-character boxes restricted to F/O/X.
    image_to_boxes returns boxes in a bottom-left origin coordinate system.
    We'll convert to OpenCV (top-left origin).
    """
    h_img, w_img = bw.shape[:2]
    config = r"-c tessedit_char_whitelist=FOX --psm 6 --oem 1"
    boxes_str = pytesseract.image_to_boxes(bw, config=config)

    char_boxes: List[CharBox] = []
    for line in boxes_str.splitlines():
        # Format: char x1 y1 x2 y2 page
        parts = line.split()
        if len(parts) >= 5:
            ch = parts[0].upper()
            if ch not in ALLOWED:
                continue
            x1, y1, x2, y2 = map(int, parts[1:5])
            # Convert Tesseract coords (origin bottom-left) to OpenCV (origin top-left)
            y1_cv = h_img - y2
            y2_cv = h_img - y1
            x1_cv, x2_cv = x1, x2
            w = x2_cv - x1_cv
            h = y2_cv - y1_cv
            char_boxes.append(
                CharBox(ch, x1_cv, y1_cv, x2_cv, y2_cv, x1_cv + w // 2, y1_cv + h // 2, w, h)
            )
    return char_boxes

def group_rows(char_boxes: List[CharBox]) -> List[List[CharBox]]:
    """
    Cluster characters into rows using y centers with a tolerance based on median height.
    """
    if not char_boxes:
        return []
    heights = [cb.h for cb in char_boxes]
    med_h = np.median(heights) if heights else 20
    tol = max(6, int(0.6 * med_h))  # vertical grouping tolerance

    # Sort by y center (top → bottom)
    sorted_boxes = sorted(char_boxes, key=lambda c: c.yc)
    rows: List[List[CharBox]] = []
    current: List[CharBox] = []
    current_y = None

    for cb in sorted_boxes:
        if current_y is None:
            current = [cb]
            current_y = cb.yc
        else:
            if abs(cb.yc - current_y) <= tol:
                current.append(cb)
                # Update average y of row for stability
                current_y = int(np.mean([c.yc for c in current]))
            else:
                rows.append(sorted(current, key=lambda c: c.xc))
                current = [cb]
                current_y = cb.yc
    if current:
        rows.append(sorted(current, key=lambda c: c.xc))
    return rows

def within(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol

def find_horizontal(rows: List[List[CharBox]]) -> List[FoxHit]:
    hits: List[FoxHit] = []
    for row in rows:
        if len(row) < 3:
            continue
        # Estimate typical gap to help ensure adjacency in the grid (avoid "F...O...X")
        widths = [c.w for c in row]
        med_w = np.median(widths) if widths else 20
        # Tolerance allows for proportional fonts / slight spacing differences
        gap_tol = med_w * 0.9

        for i in range(len(row) - 2):
            a, b, c = row[i], row[i + 1], row[i + 2]
            if a.ch == 'F' and b.ch == 'O' and c.ch == 'X':
                gap1 = b.xc - a.xc
                gap2 = c.xc - b.xc
                if within(gap1, gap2, gap_tol):
                    x1 = min(a.x1, b.x1, c.x1)
                    y1 = min(a.y1, b.y1, c.y1)
                    x2 = max(a.x2, b.x2, c.x2)
                    y2 = max(a.y2, b.y2, c.y2)
                    hits.append(FoxHit('H', (x1, y1, x2, y2), [a, b, c]))
    return hits

def index_by_xy(char_boxes: List[CharBox]):
    """
    Build quick lookup lists to find nearest neighbors in vertical/diagonal directions.
    """
    by_x = sorted(char_boxes, key=lambda c: c.xc)
    by_y = sorted(char_boxes, key=lambda c: c.yc)
    return by_x, by_y

def find_neighbors(base: CharBox, chars: List[CharBox], dx: float, dy: float,
                   tol_x: float, tol_y: float, expect: str) -> List[CharBox]:
    """
    Find characters near (base.xc+dx, base.yc+dy) within tolerances, with matching label.
    Returns list (usually 0 or a few).
    """
    target_x = base.xc + dx
    target_y = base.yc + dy
    out = []
    for c in chars:
        if c.ch != expect:
            continue
        if within(c.xc, target_x, tol_x) and within(c.yc, target_y, tol_y):
            out.append(c)
    return out

def find_non_horizontal(char_boxes: List[CharBox]) -> List[FoxHit]:
    """
    Vertical and diagonal search using neighbor matching from each 'F'.
    """
    hits: List[FoxHit] = []
    if not char_boxes:
        return hits

    heights = [c.h for c in char_boxes]
    widths = [c.w for c in char_boxes]
    med_h = np.median(heights) if heights else 20
    med_w = np.median(widths) if widths else 20

    # Expected step between letters in grid
    step_y = med_h * 1.2
    step_x = med_w * 1.2

    tol_x = med_w * 0.7
    tol_y = med_h * 0.7

    # Directions: vertical down, diag down-right, diag down-left
    directions = [
        ('V', 0, step_y),
        ('D1', step_x, step_y),
        ('D2', -step_x, step_y),
    ]

    # Speed: restrict search set by letter
    letters_by = {'F': [], 'O': [], 'X': []}
    for c in char_boxes:
        letters_by[c.ch].append(c)

    for f in letters_by['F']:
        for label, dx, dy in directions:
            # Find 'O' near the next step
            os = find_neighbors(f, letters_by['O'], dx, dy, tol_x, tol_y, 'O')
            for o in os:
                # Then 'X' one more step in the same direction
                xs = find_neighbors(o, letters_by['X'], dx, dy, tol_x, tol_y, 'X')
                for x in xs:
                    x1 = min(f.x1, o.x1, x.x1)
                    y1 = min(f.y1, o.y1, x.y1)
                    x2 = max(f.x2, o.x2, x.x2)
                    y2 = max(f.y2, o.y2, x.y2)
                    hits.append(FoxHit(label, (x1, y1, x2, y2), [f, o, x]))

    return hits

def annotate_and_save(original_path: str, out_dir: str, hits: List[FoxHit]) -> str:
    img = cv2.imread(original_path)
    if img is None:
        return ""
    for hit in hits:
        x1, y1, x2, y2 = hit.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(img, f"FOX-{hit.direction}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2, cv2.LINE_AA)
    base = os.path.basename(original_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}_FOX{ext}")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path

def process_image(path: str, annotate_dir: str = None) -> Dict:
    bw = load_and_binarize(path)
    boxes = ocr_char_boxes(bw)
    rows = group_rows(boxes)

    horizontal_hits = find_horizontal(rows)
    other_hits = find_non_horizontal(boxes)

    # Deduplicate overlapping hits (same bbox area overlap > 70%)
    all_hits = horizontal_hits + other_hits
    deduped: List[FoxHit] = []
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    for h in all_hits:
        keep = True
        for k in deduped:
            if iou(h.bbox, k.bbox) > 0.7:
                keep = False
                break
        if keep:
            deduped.append(h)

    out_annot = None
    if deduped and annotate_dir:
        out_annot = annotate_and_save(path, annotate_dir, deduped)

    result = {
        "image": path,
        "found": bool(deduped),
        "count": len(deduped),
        "directions": [h.direction for h in deduped],
        "annotated_image": out_annot
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Find the page that contains the word FOX.")
    parser.add_argument("--input", required=True, help="Folder with page images (png/jpg/jpeg).")
    parser.add_argument("--output", default="fox_hits", help="Folder to save annotated pages.")
    parser.add_argument("--no-annotate", action="store_true", help="Do not write annotated images.")
    args = parser.parse_args()

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    paths = [
        os.path.join(args.input, f)
        for f in sorted(os.listdir(args.input))
        if os.path.splitext(f.lower())[1] in exts
    ]

    if not paths:
        print("No images found in input folder.")
        return

    print(f"Scanning {len(paths)} images for 'FOX' ...")
    found_any = False
    for p in paths:
        res = process_image(p, None if args.no_annotate else args.output)
        if res["found"]:
            found_any = True
            print(f"[HIT] {os.path.basename(p)} -> {res['count']} occurrence(s), directions={res['directions']}")
            if res["annotated_image"]:
                print(f"      Annotated: {res['annotated_image']}")
        else:
            print(f"[   ] {os.path.basename(p)} -> no 'FOX' detected")

    if not found_any:
        print("No pages with 'FOX' were detected. Try adjusting image quality or OCR settings.")
        print("Tips: ensure sharp scans, higher DPI, and good contrast. If the book font is stylized,")
        print("      consider scanning in grayscale and increasing contrast, or set --psm 11 in config.")

if __name__ == "__main__":
    main()
