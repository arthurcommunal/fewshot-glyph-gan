import numpy as np

from glyphgan.drawing import canvas_to_px


def _canvas(dy: int = 0, dx: int = 0, scale: float = 1.0, size: int = 420, width: int = 14):
    """RGBA canvas, transparent background, a black 'T' drawn at a given offset and scale."""
    c = np.zeros((size, size, 4), np.uint8)
    y0, x0 = 80 + dy, 100 + dx
    L = int(200 * scale)
    c[y0:y0 + width, x0:x0 + L] = (0, 0, 0, 255)
    c[y0:y0 + L, x0 + L // 2 - width // 2:x0 + L // 2 + width // 2] = (0, 0, 0, 255)
    return c


def test_empty_canvas():
    assert canvas_to_px(np.zeros((420, 420, 4), np.uint8)).max() == 0


def test_output_format():
    px = canvas_to_px(_canvas())
    assert px.shape == (28, 28) and px.dtype == np.uint8 and px.max() == 255
    assert px[:3].max() == 0 and px[-3:].max() == 0          # glyph inside the centred 22x22 box


def test_invariant_to_position_and_size():
    ref = canvas_to_px(_canvas()).astype(float)
    for c in (_canvas(dy=90, dx=-60), _canvas(scale=0.6), _canvas(scale=0.6, width=9)):
        assert np.abs(canvas_to_px(c).astype(float) - ref).mean() / 255 < 0.03


def test_opaque_white_background():
    c = _canvas()
    c[..., :3][c[..., 3] == 0] = 255
    c[..., 3] = 255
    assert np.array_equal(canvas_to_px(c), canvas_to_px(_canvas()))
