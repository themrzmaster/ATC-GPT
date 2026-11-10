"""
Remove X-Plane's own UI from a screenshot before the model sees it: the notification boxes stacked in the
top-right ("scenery package failed to load", "rendering options too high", ...) and the PAUSED/toolbar strip.
A pilot never sees these; in fog the model took a "failed to load" box as proof the airport didn't exist.

We can't dismiss them in X-Plane (no command for it) and the X-Plane install is off limits, so they are painted
over with the view around them: each row of a box is filled by blending the colours just left and right of it.
"""

import numpy as np
from PIL import Image, ImageFilter

# X-Plane 12 notification background: flat neutral dark grey (~RGB 21,25,26), unlike the sky/fog and unlike the
# near-black glareshield below it
_BOX_LO, _BOX_HI, _BOX_NEUTRAL = 8, 40, 14
_COLS = (0.775, 0.98)        # where the boxes sit, as a fraction of the width
_MAX_Y = 0.75                # boxes never go below this (the panel is there)
_ROW_FRAC = 0.45             # a box row is mostly background even with its text and buttons
_MERGE_PX = 14               # stacked boxes: close the small gaps between them
_TOOLBAR = (0.86, 0.035)     # PAUSED + toolbar icons: x from 86% of the width, top 3.5% of the height


def _box_mask(a: np.ndarray) -> np.ndarray:
    mx, mn = a.max(axis=2), a.min(axis=2)
    return (mn >= _BOX_LO) & (mx <= _BOX_HI) & (mx - mn <= _BOX_NEUTRAL)


def _runs(flags: np.ndarray, merge: int) -> list[tuple[int, int]]:
    runs, start, last = [], None, None
    for i in np.flatnonzero(flags):
        if start is None:
            start = last = i
        elif i - last <= merge:
            last = i
        else:
            runs.append((start, last))
            start = last = i
    if start is not None:
        runs.append((start, last))
    return runs


def hide_sim_overlays(im: Image.Image) -> tuple[Image.Image, int]:
    """(cleaned image, number of notification boxes painted over)."""
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    h, w, _ = a.shape
    out = a.copy()
    mask = _box_mask(a)
    x0c, x1c = int(_COLS[0] * w), int(_COLS[1] * w)
    ylim = int(_MAX_Y * h)
    # a box floats on the view: the strip right of it is sky/scenery, whereas the dark panel runs to the edge
    margin = a[:ylim, int(0.992 * w):].mean(axis=(1, 2))
    rows = (mask[:ylim, x0c:x1c].mean(axis=1) > _ROW_FRAC) & (margin > 45)
    boxes, patches = 0, []
    for y0, y1 in _runs(rows, _MERGE_PX):
        if y1 - y0 < 12:                     # a real box is tall; ignore stray dark rows
            continue
        cols = mask[y0:y1 + 1].mean(axis=0) > 0.5
        xs = np.flatnonzero(cols[int(0.6 * w):]) + int(0.6 * w)
        if not len(xs):
            continue
        bx0, bx1 = max(xs.min() - 6, 0), min(xs.max() + 6, w - 1)
        by0, by1 = max(y0 - 6, 0), min(y1 + 6, h - 1)
        for y in range(by0, by1 + 1):
            left = np.median(a[y, max(bx0 - 12, 0):max(bx0 - 2, 1)], axis=0)
            right = np.median(a[y, min(bx1 + 3, w - 1):w], axis=0) if bx1 + 3 < w else left
            t = np.linspace(0.0, 1.0, bx1 - bx0 + 1)[:, None]
            out[y, bx0:bx1 + 1] = (left * (1 - t) + right * t).astype(np.int16)
        patches.append((bx0, by0, bx1 + 1, by1 + 1))
        boxes += 1
    tx, ty = int(_TOOLBAR[0] * w), max(int(_TOOLBAR[1] * h), 1)
    out[:ty, tx:] = out[ty + 2, tx:][None, :, :]          # continue the sky just below the strip upwards
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    for box in patches:                       # soften the row-by-row fill so it reads as haze, not streaks
        img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(5)), box[:2])
    return img, boxes
