"""Shared ink-detection threshold constants.

Two distinct concepts live here so they are never silently re-invented in
individual modules:

INK_BBOX_THRESHOLD (250)
    Used for *bounding-box refit* after rotation (geometry.py).  It must be
    generous: INTER_CUBIC warpAffine produces sub-pixel anti-aliasing halos
    that are visually white but numerically < 255.  Raising the threshold to
    250 captures those edge pixels so the tight bbox does not clip ink.

INK_BINARIZE_THRESHOLD (165)
    Used for *hard binarisation* of glyph tiles after BICUBIC resize
    (synth_pool.py).  It must be tighter than INK_BBOX_THRESHOLD: BICUBIC
    produces blurry grey halos around strokes that, if kept, would produce
    artificially thin or faint glyphs in the synthetic scenes.  165 cuts the
    halo while preserving stroke body -- this is also the boldness-regression
    fix target (audit accf9872545875b82).

INK_DRAW_THRESHOLD (200)
    Used for *stroke detection* in drawn elements (strokes.py).  A legacy
    value retained because stroke pixels are painted with cv2.line at value 0
    on a 255-background; any pixel < 200 reliably indicates ink there.

Drift warning: do not copy these values inline.  Import from here.
"""

# Pixels strictly below this threshold are treated as ink for bbox refit.
INK_BBOX_THRESHOLD: int = 250

# Pixels strictly below this threshold are binarised to black ink after resize.
INK_BINARIZE_THRESHOLD: int = 200

# Pixels strictly below this threshold are considered drawn ink in stroke paths.
INK_DRAW_THRESHOLD: int = 200
