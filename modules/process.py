def _names(detections):
    """Class-name array straight from the detections object."""
    return detections.data["class_name"]


def _center(box):
    """Center (x, y) of a box — works on a 4-tuple (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def find_class_center(detections, target):
    """Return (cx, cy) of the first detection whose class_name == target, or None."""
    names = _names(detections)
    for i, name in enumerate(names):
        if name == target:
            x1, y1, x2, y2 = detections.xyxy[i]
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return None


def absolute_coords(
    detections,
    units_per_span=10.0,
    keep=(
        "ball",
        "full_tank",
        "car",
        "boost",
        "nametag",
        "goal",
        "joystick",
        "jump_avail",
        "jump_unavail",
        "boost_avail",
        "boost_unavail",
    ),
):
    ref = find_class_center(detections, "ref1")
    ref2 = find_class_center(detections, "ref2")
    if ref is None or ref2 is None:
        return None
    rx, ry = ref
    span = ((ref2[0] - rx) ** 2 + (ref2[1] - ry) ** 2) ** 0.5
    if span == 0:
        return None
    upp = units_per_span / span

    names = _names(detections)
    out = {}
    for i, name in enumerate(names):
        if name not in keep:
            continue
        x1, y1, x2, y2 = detections.xyxy[i]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ux = (cx - rx) * upp
        uy = (ry - cy) * upp
        out.setdefault(name, []).append((ux, uy, cx, cy))
    return out


def draw_coord_system(
    frame,
    detections,
    cleaned=None,
    video="/923-1/",
    units_per_span=5.0,
    save=True,
):
    """
    Overlay a ref-origin coordinate grid + detection boxes and save it.
    Origin = ref center. Scale = ref->ref2 pixel span (cancels camera offset).
    Square units, +y up. Returns annotated BGR image, or None if a marker is missing.
    """

    import os
    import cv2
    import numpy as np
    from datetime import datetime

    ref = find_class_center(detections, "ref1")
    ref2 = find_class_center(detections, "ref2")
    if ref is None or ref2 is None:
        print("ref or ref2 not detected — no grid drawn")
        return None

    h, w = frame.shape[:2]
    rx, ry = ref
    span = ((ref2[0] - rx) ** 2 + (ref2[1] - ry) ** 2) ** 0.5
    if span == 0:
        print("ref and ref2 coincide — no scale")
        return None

    ppu = span / units_per_span  # pixels per unit
    upp = units_per_span / span  # units per pixel

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()

    grid_color = (90, 90, 90)
    axis_color = (0, 165, 255)  # orange axes (BGR)
    label_color = (255, 255, 255)
    box_color = (0, 220, 0)  # green boxes
    center_color = (0, 0, 255)  # red centers/labels
    ref_color = (255, 200, 0)  # cyan-ish for the markers

    # --- grid: vertical lines at each integer x-unit ---
    max_ux = int(np.ceil((w - rx) * upp))
    min_ux = int(np.floor((0 - rx) * upp))
    for ux in range(min_ux, max_ux + 1):
        px = int(rx + ux * ppu)
        if 0 <= px < w:
            color = axis_color if ux == 0 else grid_color
            thick = 2 if ux == 0 else 1
            cv2.line(img, (px, 0), (px, h), color, thick)
            if ux != 0:
                cv2.putText(
                    img,
                    f"{ux:+d}",
                    (px + 2, int(ry) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    label_color,
                    1,
                )

    # --- grid: horizontal lines at each integer y-unit (+y up) ---
    max_uy = int(np.ceil(ry * upp))
    min_uy = int(np.floor((ry - h) * upp))
    for uy in range(min_uy, max_uy + 1):
        py = int(ry - uy * ppu)
        if 0 <= py < h:
            color = axis_color if uy == 0 else grid_color
            thick = 2 if uy == 0 else 1
            cv2.line(img, (0, py), (w, py), color, thick)
            if uy != 0:
                cv2.putText(
                    img,
                    f"{uy:+d}",
                    (int(rx) + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    label_color,
                    1,
                )

    # --- origin + both markers ---
    cv2.circle(img, (int(rx), int(ry)), 4, axis_color, -1)
    cv2.putText(
        img,
        "(0,0) ref",
        (int(rx) + 6, int(ry) + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        axis_color,
        1,
    )
    cv2.circle(img, (int(ref2[0]), int(ref2[1])), 4, ref_color, -1)
    cv2.putText(
        img,
        "ref2",
        (int(ref2[0]) + 6, int(ref2[1]) + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        ref_color,
        1,
    )

    # --- detections: box + center + unit coordinate ---
    names = detections.data["class_name"]
    for i, name in enumerate(names):
        # if name not in keep:
        #     continue
        x1, y1, x2, y2 = [int(v) for v in detections.xyxy[i]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ux = (cx - rx) * upp
        uy = (ry - cy) * upp

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            img, name, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1
        )
        cv2.circle(img, (int(cx), int(cy)), 4, center_color, -1)
        cv2.putText(
            img,
            f"({ux:+.1f},{uy:+.1f})",
            (int(cx) + 6, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            center_color,
            1,
        )

        # --- cleaned objects: blue boxes (overlay on raw green) ---
    if cleaned is not None:
        _draw_cleaned_boxes(img, cleaned, rx, ry, upp)

    if cleaned is not None:
        _draw_owned_dots(img, cleaned)

    if save:
        out_dir = f"images/coordinatesystem{video}"
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        path = os.path.join(out_dir, f"coords_{ts}.png")
        cv2.imwrite(path, img)
        print(f"saved {path}")

    return img


def _draw_owned_dots(img, cleaned):
    """Yellow dots on yours (my_car, my_goal), pink on opponent's (opp_car, opp_goal)."""
    import cv2

    YELLOW = (0, 255, 255)  # BGR
    PINK = (203, 192, 255)  # BGR
    WHITE = (255, 255, 255)  # BGR

    def dot(obj, color):
        if obj is None:
            return
        cx, cy = int(obj[4]), int(obj[5])  # pixel center = tuple indices 4,5
        cv2.circle(img, (cx, cy), 10, color, -1)
        cv2.circle(img, (cx, cy), 10, (0, 0, 0), 2)  # black outline for visibility

    dot(cleaned.get("my_car"), YELLOW)
    dot(cleaned.get("opp_car"), PINK)
    dot(cleaned.get("ball")[0], WHITE)
    for g in cleaned.get("goal", []):
        dot(g, YELLOW if g[0] >= 0 else PINK)  # +x side = yours


def find_class_center(detections, target):
    """Return (cx, cy) of the first detection whose class_name == target, or None."""
    names = _names(detections)
    for i, name in enumerate(names):
        if name == target:
            x1, y1, x2, y2 = detections.xyxy[i]
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return None


def _draw_cleaned_boxes(img, cleaned, rx, ry, upp, half=40):
    """
    Draw BLUE boxes for cleaned objects (post edit_tracked), overlaying the
    green raw-detection boxes. Cleaned tuples carry pixel center (cx,cy) at
    indices 4,5 but no corners, so we draw a fixed half-size box around the
    center. Shows where cleaning placed each object (incl. reconstructed ones).
    """
    BLUE = (255, 100, 0)  # BGR (blue-ish)

    import cv2

    def box(obj, label):
        if obj is None:
            return
        cx, cy = int(obj[4]), int(obj[5])
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), BLUE, 2)
        cv2.putText(
            img,
            label,
            (cx - half, cy - half - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            BLUE,
            1,
        )
        # cleaned unit coordinate
        ux = (cx - rx) * upp
        uy = (ry - cy) * upp
        cv2.putText(
            img,
            f"({ux:+.1f},{uy:+.1f})",
            (cx + 6, cy + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            BLUE,
            1,
        )

    # cars by explicit key
    box(cleaned.get("my_car"), "my_car*")
    box(cleaned.get("opp_car"), "opp_car*")

    # ball
    balls = cleaned.get("ball") or []
    if balls:
        box(balls[0], "ball*")

    # goals
    for g in cleaned.get("goal", []):
        box(g, "my_goal*" if g[0] >= 0 else "opp_goal*")

    # other single-instance cleaned objects with pixel centers
    for key in ("nametag", "boost_indicator"):
        items = cleaned.get(key)
        if items:
            box(items[0], key + "*")
