"""VDSP v1.9 물리 계층 테스트 패턴: 레이아웃과 프레임 생성.

vdsp_frame.js와 비트 단위로 같은 결과를 내야 한다 (selftest.py가 비교).
셀 값: 0=청(B) 1=적(R) 2=녹(G) 3=백(W)  (spec 2.1)
"""
import numpy as np

S = 192
BLUE, RED, GREEN, WHITE = 0, 1, 2, 3
COLOR_NAMES = ("B", "R", "G", "W")
PALETTE_RGB = np.array([[0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 255]], np.uint8)

# 셀 값(역할) 0~3 → 표시 색. 역할은 휘도 순서(0이 가장 어두움)라서 앵커·핀·분리대·테두리의
# 명암 관계가 팔레트가 바뀌어도 같은 방식으로 유지된다.
PALETTES = {
    "rgbw": (PALETTE_RGB, COLOR_NAMES),                      # spec 2.1
    "cmyg": (np.array([[255, 0, 255], [0, 255, 0], [0, 255, 255], [255, 255, 0]], np.uint8),
             ("M", "G", "C", "Y")),                          # libcimbar 4색과 같은 색 (휘도 순 배치)
}


def palette_rgb(name="rgbw"):
    return PALETTES[name][0]


def color_names(name="rgbw"):
    return PALETTES[name][1]
DATA, BORDER, ANCHOR, SEPARATOR, PIN = 0, 1, 2, 3, 4
ANCHOR_ORIGINS = ((3, 3), (181, 3), (3, 181), (181, 181))  # TL, TR, BL, BR
ANCHOR_PATTERN = ((BLUE, GREEN), (WHITE, RED))  # [qy][qx]
GRAY2 = (0, 1, 3, 2)
M32 = 0xFFFFFFFF
_SHIFTS = np.arange(30, -1, -2, dtype=np.uint32)


def _in_anchor(c):
    return 3 <= c <= 10 or 181 <= c <= 188


def _in_sep(c):
    return 2 <= c <= 11 or 180 <= c <= 189


def _anchor_origin(c):
    return 3 if c <= 11 else 181


def anchor_color(ax, ay):
    return ANCHOR_PATTERN[1 if ay >= 4 else 0][1 if ax >= 4 else 0]


def pin_color(dx, dy, style):
    if style == "cross":
        return GREEN if dx == 0 or dy == 0 else BLUE
    return GREEN if (dx + dy) % 2 == 0 else BLUE  # spec 2.4 체커보드


class Layout:
    def __init__(self, pin_style="checker"):
        if pin_style not in ("checker", "cross"):
            raise ValueError(f"unknown pin style: {pin_style}")
        self.pin_style = pin_style
        region = np.zeros((S, S), np.uint8)
        fixed = np.full((S, S), -1, np.int8)
        ring = np.full((S, S), -1, np.int8)
        ring_idx = np.full((S, S), -1, np.int32)

        for y in range(S):
            for x in range(S):
                if x < 2 or y < 2 or x >= S - 2 or y >= S - 2:
                    region[y, x] = BORDER
                    k = min(x, y, S - 1 - x, S - 1 - y)
                    last, L = S - 1 - k, S - 2 * k
                    if y == k:
                        i = x - k
                    elif x == last:
                        i = (L - 1) + (y - k)
                    elif y == last:
                        i = 2 * (L - 1) + (last - x)
                    else:
                        i = 3 * (L - 1) + (last - y)
                    ring[y, x], ring_idx[y, x] = k, i
                elif _in_anchor(x) and _in_anchor(y):
                    region[y, x] = ANCHOR
                    fixed[y, x] = anchor_color(x - _anchor_origin(x), y - _anchor_origin(y))
                elif _in_sep(x) and _in_sep(y):
                    region[y, x] = SEPARATOR
                    ox, oy = _anchor_origin(x), _anchor_origin(y)
                    c = anchor_color(min(max(x, ox), ox + 7) - ox, min(max(y, oy), oy + 7) - oy)
                    fixed[y, x] = WHITE if c in (BLUE, RED) else BLUE
                else:
                    i, j = int(np.floor(x / 16 + 0.5)), int(np.floor(y / 16 + 0.5))
                    if 1 <= i <= 11 and 1 <= j <= 11 and abs(x - 16 * i) <= 1 and abs(y - 16 * j) <= 1:
                        region[y, x] = PIN
                        fixed[y, x] = pin_color(x - 16 * i, y - 16 * j, pin_style)

        self.region, self.fixed, self.ring, self.ring_idx = region, fixed, ring, ring_idx
        counts = [int((region == r).sum()) for r in (BORDER, ANCHOR, SEPARATOR, PIN, DATA)]
        assert counts == [1520, 256, 144, 1089, 33855], counts  # spec 2.6

        # 래스터 순서 (np.nonzero는 행 우선)
        self.data_flat = np.flatnonzero(region.ravel() == DATA)

        # 테두리 셀 쌍: 링별로 번호순 (0,1), (2,3), ...
        first, second = [], []
        for k in (0, 1):
            cells = np.flatnonzero(ring.ravel() == k)
            cells = cells[np.argsort(ring_idx.ravel()[cells])]
            first.append(cells[0::2])
            second.append(cells[1::2])
        self.pair_first = np.concatenate(first)
        self.pair_second = np.concatenate(second)


def frame_seed(idx, seed):
    s = ((0x9E3779B9 * (idx + 1)) & M32) ^ (seed & M32)
    return s or 1


def _xorshift(x):
    x ^= (x << 13) & M32
    x ^= x >> 17
    x ^= (x << 5) & M32
    return x


def counter_value(idx, mode="bin"):
    return GRAY2[idx & 3] if mode == "gray" else idx & 3


def gen_frame(layout, idx, seed=1, counter="bin"):
    """프레임 idx의 셀 값 (S, S) uint8."""
    cells = np.zeros(S * S, np.uint8)
    fixed = layout.fixed.ravel()
    cells[fixed >= 0] = fixed[fixed >= 0]

    ctr = counter_value(idx, counter)
    ring = layout.ring.ravel()
    border = ring >= 0
    bit = np.where(ring == 1, ctr & 1, (ctr >> 1) & 1)
    first = layout.ring_idx.ravel() % 2 == 0
    cells[border] = np.where((bit == 1) == first, WHITE, BLUE)[border]

    n = len(layout.data_flat)
    x = frame_seed(idx, seed)
    for _ in range(8):
        x = _xorshift(x)
    words = []
    for _ in range(-(-n // 16)):
        x = _xorshift(x)
        words.append(x)
    w = np.array(words, np.uint32)
    vals = ((w[:, None] >> _SHIFTS[None, :]) & 3).ravel()[:n]
    cells[layout.data_flat] = vals
    return cells.reshape(S, S)


def render(cells, px, quiet=4, palette="rgbw"):
    """셀 격자를 RGB 이미지로 (흑색 여백 quiet셀 포함)."""
    img = palette_rgb(palette)[cells]
    img = np.pad(img, ((quiet, quiet), (quiet, quiet), (0, 0)))
    return np.repeat(np.repeat(img, px, 0), px, 1)
