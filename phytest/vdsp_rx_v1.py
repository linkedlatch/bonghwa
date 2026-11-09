"""VDSP 물리 계층 테스트 수신기: 검출, 기하 보정, 색 판정, 채점.

좌표계: (u, v) = 셀 단위 프레임 좌표. 셀 (x, y)는 [x, x+1] × [y, y+1]을 차지하고
중심은 (x+0.5, y+0.5). 이미지 좌표는 픽셀 중심이 정수.
"""
import numpy as np
import cv2

from vdsp_layout import (S, BLUE, GREEN, WHITE, DATA, BORDER, ANCHOR_ORIGINS, ANCHOR_PATTERN,
                         Layout, gen_frame, pin_color, palette_rgb, color_names)

SUB = np.array([-0.2, 0.0, 0.2])          # 셀 안 표본 위치 (셀 중앙 약 40% 영역)
LUMA = np.array([0.2126, 0.7152, 0.0722])  # BT.709
NFEAT = {0: 0, 1: 3, 2: 6, 3: 10}


class Context:
    """테스트 시퀀스 정보. 송신기의 seed / n / pin / counter와 같아야 한다."""

    def __init__(self, seed=1, n=64, pin="checker", counter="bin", palette="rgbw"):
        self.palette = palette
        self.color_names = color_names(palette)
        self.ideal = palette_rgb(palette) / 255.0
        self.layout = L = Layout(pin)
        self.n = n
        self.expected = np.stack([gen_frame(L, i, seed, counter) for i in range(n)])
        self.data_mask = L.region == DATA
        self.dflat = L.data_flat
        self.exp_data = self.expected.reshape(n, -1)[:, self.dflat]
        self.fixed_mask = L.fixed >= 0
        self.pin_uv = np.array([(16 * i + 0.5, 16 * j + 0.5)
                                for j in range(1, 12) for i in range(1, 12)], float)
        offs = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        self.pin_cell_off = np.array(offs, float)
        green = np.array([pin_color(dx, dy, pin) == GREEN for dx, dy in offs])
        self.pin_green = green
        # 평균 0 템플릿: 전체 밝기 변화에 영향받지 않게
        self.pin_tmpl = np.where(green, 1.0 / green.sum(), -1.0 / (~green).sum())


# ---------------------------------------------------------------- 표본 추출

def sample(img, xs, ys):
    """이미지를 임의 모양의 좌표 배열에서 쌍선형 보간."""
    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)
    shape, n = xs.shape, xs.size
    cols = 2048
    rows = max(1, -(-n // cols))
    pad = rows * cols - n
    mx = np.pad(xs.ravel(), (0, pad)).reshape(rows, cols)
    my = np.pad(ys.ravel(), (0, pad)).reshape(rows, cols)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if img.ndim == 3:
        return out.reshape(rows * cols, img.shape[2])[:n].reshape(shape + (img.shape[2],))
    return out.reshape(-1)[:n].reshape(shape)


def poly_feats(u, v, order):
    s = (np.asarray(u, float) - S / 2) / (S / 2)
    t = (np.asarray(v, float) - S / 2) / (S / 2)
    f = [np.ones_like(s), s, t]
    if order >= 2:
        f += [s * s, s * t, t * t]
    if order >= 3:
        f += [s ** 3, s * s * t, s * t * t, t ** 3]
    return np.stack(f, -1)


class Mapping:
    """셀 좌표 → 이미지 좌표: 호모그래피 + 다항식 보정 + 핀 격자 잔차 보간."""

    def __init__(self, H):
        self.H = np.asarray(H, float)
        self.order = 0
        self.coef = None
        self.resid = None  # (11, 11, 2) 이미지 px, [j, i]

    def add_poly(self, order, coef):
        o = max(order, self.order)
        c = np.zeros((NFEAT[o], 2))
        if self.coef is not None:
            c[:NFEAT[self.order]] += self.coef
        c[:NFEAT[order]] += coef
        self.order, self.coef = o, c

    def __call__(self, u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        H = self.H
        w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
        x = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / w
        y = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / w
        if self.coef is not None:
            F = poly_feats(u, v, self.order)
            x = x + F @ self.coef[:, 0]
            y = y + F @ self.coef[:, 1]
        if self.resid is not None:
            gi = np.clip((u - 0.5) / 16 - 1, 0, 10)
            gj = np.clip((v - 0.5) / 16 - 1, 0, 10)
            i0 = np.minimum(gi.astype(int), 9)
            j0 = np.minimum(gj.astype(int), 9)
            fi, fj = (gi - i0)[..., None], (gj - j0)[..., None]
            R = self.resid
            r = (R[j0, i0] * (1 - fi) * (1 - fj) + R[j0, i0 + 1] * fi * (1 - fj)
                 + R[j0 + 1, i0] * (1 - fi) * fj + R[j0 + 1, i0 + 1] * fi * fj)
            x = x + r[..., 0]
            y = y + r[..., 1]
        return x, y


# ---------------------------------------------------------------- 검출

def detect_quad(rgb):
    """흑색 여백으로 둘러싸인 밝은 사각형(프레임 외곽)의 네 꼭짓점. 시계 방향."""
    V = (rgb.max(axis=2) * 255).astype(np.uint8)  # 순색 4가지는 모두 max 채널이 밝다
    blur = cv2.GaussianBlur(V, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = 0.01 * V.size
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        hull = cv2.convexHull(cnt)
        if area < 0.85 * cv2.contourArea(hull):
            continue
        peri = cv2.arcLength(hull, True)
        for eps in np.linspace(0.005, 0.08, 30):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) <= 4:
                break
        if len(ap) != 4:
            continue
        pts = ap.reshape(4, 1, 2).astype(np.float32)
        cv2.cornerSubPix(blur, pts, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        pts = pts.reshape(4, 2)
        c = pts.mean(0)
        order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        return pts[order].astype(np.float64)
    return None


def _anchor_quadrant_points():
    """앵커 4개 × 구역 4개의 표본 좌표 (4, 4, 25, 2)와 각 구역의 기대 색."""
    g = np.linspace(-1, 1, 5)
    gu, gv = np.meshgrid(g, g)
    pts = np.zeros((4, 4, 25, 2))
    cols = np.zeros((4, 4), int)
    for a, (ox, oy) in enumerate(ANCHOR_ORIGINS):
        for q, (qx, qy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            pts[a, q, :, 0] = ox + 4 * qx + 2 + gu.ravel()
            pts[a, q, :, 1] = oy + 4 * qy + 2 + gv.ravel()
            cols[a, q] = ANCHOR_PATTERN[qy][qx]
    return pts, cols


_AQ_PTS, _AQ_COLS = _anchor_quadrant_points()
_IDEAL = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]], float)


def orient(rgb, quad, ideal=_IDEAL):
    """8가지 (회전 × 거울상) 가설 중 앵커 색이 가장 잘 맞는 호모그래피."""
    C = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
    results = []
    for mirror in (0, 1):
        for r in range(4):
            idx = [(i + r) % 4 for i in range(4)] if not mirror else [(r - i) % 4 for i in range(4)]
            H = cv2.getPerspectiveTransform(C, np.float32(quad[idx]))
            mp = Mapping(H)
            x, y = mp(_AQ_PTS[..., 0], _AQ_PTS[..., 1])
            s = sample(rgb, x, y).mean(2)                       # (4 앵커, 4 구역, 3)
            s = s / (s.max(axis=1, keepdims=True) + 1e-6)       # 앵커별 채널 정규화
            score = ((s - ideal[_AQ_COLS]) ** 2).sum()
            results.append((score, H, r, mirror))
    results.sort(key=lambda t: t[0])
    best, second = results[0], results[1]
    return best[1], best[2], best[3], second[0] / max(best[0], 1e-9)


# ---------------------------------------------------------------- 정렬 핀

def refine_pins(Y, mp, ctx, rng, step, ideal):
    """핀마다 (du, dv) 오프셋을 격자 탐색. 휘도(Y)에서 핀 템플릿과 상관.
    크로마(G−B)는 4:2:0 서브샘플링에서 1셀 주기 체커보드가 지워지므로 쓰지 않는다."""
    offs = np.arange(-rng, rng + step / 2, step)
    O = len(offs)
    dv_g, du_g = np.meshgrid(offs, offs, indexing="ij")
    du, dv = du_g.ravel(), dv_g.ravel()
    su, sv = [a.ravel() for a in np.meshgrid(SUB, SUB)]
    pc, co = ctx.pin_uv, ctx.pin_cell_off
    # 기준점은 정확히 변환하고, 탐색 오프셋은 핀 중심의 야코비안으로 더한다 (±1.5셀 안에서 선형 근사)
    bx, by = mp(pc[:, 0, None, None] + co[None, :, 0, None] + su,
                pc[:, 1, None, None] + co[None, :, 1, None] + sv)          # (P, 9, 9)
    cx, cy = mp(pc[:, 0], pc[:, 1])
    ux, uy = mp(pc[:, 0] + 1, pc[:, 1])
    vx, vy = mp(pc[:, 0], pc[:, 1] + 1)
    Jux, Juy, Jvx, Jvy = ux - cx, uy - cy, vx - cx, vy - cy                 # (P,)
    ox = Jux[:, None] * du + Jvx[:, None] * dv                              # (P, O*O)
    oy = Juy[:, None] * du + Jvy[:, None] * dv
    x = bx[:, None] + ox[:, :, None, None]
    y = by[:, None] + oy[:, :, None, None]
    vals = sample(Y, x, y).mean(-1)             # (P, O*O, 9)
    score = vals @ ctx.pin_tmpl                  # (P, O*O)
    P = len(pc)
    ar = np.arange(P)
    b = score.argmax(1)
    bs = score[ar, b]
    iv, iu = np.divmod(b, O)
    sc = score.reshape(P, O, O)
    in_u = (iu > 0) & (iu < O - 1)
    in_v = (iv > 0) & (iv < O - 1)
    iuc, ivc = np.clip(iu, 1, O - 2), np.clip(iv, 1, O - 2)

    def parab(m, c, p):
        den = m - 2 * c + p
        return np.clip(np.where(np.abs(den) > 1e-9, 0.5 * (m - p) / np.where(den == 0, 1, den), 0), -0.5, 0.5)

    fu = np.where(in_u, parab(sc[ar, iv, iuc - 1], sc[ar, iv, iuc], sc[ar, iv, iuc + 1]), 0)
    fv = np.where(in_v, parab(sc[ar, ivc - 1, iu], sc[ar, ivc, iu], sc[ar, ivc + 1, iu]), 0)
    off = np.stack([offs[iu] + fu * step, offs[iv] + fv * step], 1)
    ok = (bs > 0.5 * ideal) & in_u & in_v
    return off, ok, bs


def fit_displacement(uv, d, cell_px, max_order=3, tol=0.15):
    n = len(uv)
    order = min(max_order, 3 if n >= 30 else 2 if n >= 12 else 1 if n >= 4 else 0)
    if order == 0:
        return 0, None, np.zeros(n, bool)
    keep = np.ones(n, bool)
    Fall = poly_feats(uv[:, 0], uv[:, 1], order)
    for _ in range(4):
        coef = np.linalg.lstsq(Fall[keep], d[keep], rcond=None)[0]
        r = np.linalg.norm(d - Fall @ coef, axis=1)
        new = r <= max(3 * np.median(r[keep]), tol * cell_px)
        if new.sum() < NFEAT[order] + 2 or (new == keep).all():
            break
        keep = new
    return order, coef, keep


def locate(rgb, ctx, trace=None):
    """프레임을 찾아 셀 좌표 → 이미지 좌표 매핑을 만든다. 실패하면 None.
    trace에 dict를 넘기면 시각화용 중간 결과를 담는다."""
    quad = detect_quad(rgb)
    if quad is None:
        return None
    H, rot, mirror, orient_ratio = orient(rgb, quad, ctx.ideal)
    mp = Mapping(H)
    Y = np.ascontiguousarray(rgb @ LUMA.astype(np.float32))
    if trace is not None:
        trace.update(quad=quad, H=H.copy(), rot=rot, mirror=mirror, orient_ratio=orient_ratio, rounds=[])

    x, y = mp(_AQ_PTS[..., 0], _AQ_PTS[..., 1])
    q = sample(Y, x, y).mean(2)
    yG = q[_AQ_COLS == GREEN].mean()
    yB = q[_AQ_COLS == BLUE].mean()
    ideal = float((np.where(ctx.pin_green, yG, yB) * ctx.pin_tmpl).sum())

    xs, ys = mp(np.array([96.0, 97.0, 96.0]), np.array([96.0, 96.0, 97.0]))
    cell_px = 0.5 * (np.hypot(xs[1] - xs[0], ys[1] - ys[0]) + np.hypot(xs[2] - xs[0], ys[2] - ys[0]))

    pc = ctx.pin_uv
    # 거친 탐색 → 세밀한 탐색. 첫 단계는 체커보드의 대각선 1셀 오정합(P-1)이 섞일 수 있어
    # 저차 모델 + 느슨한 이상치 기준으로 걸러낸다.
    for rng, step, max_order, tol, use_resid in ((1.5, 0.25, 2, 0.4, False),
                                                 (0.75, 0.125, 3, 0.2, False),
                                                 (0.3, 0.05, 3, 0.15, True)):
        off, ok, _ = refine_pins(Y, mp, ctx, rng, step, ideal)
        if ok.sum() < 4:
            break
        x1, y1 = mp(pc[:, 0] + off[:, 0], pc[:, 1] + off[:, 1])
        x0, y0 = mp(pc[:, 0], pc[:, 1])
        d = np.stack([x1 - x0, y1 - y0], 1)
        order, coef, keep = fit_displacement(pc[ok], d[ok], cell_px, max_order, tol)
        if trace is not None:
            inlier = np.zeros(len(pc), bool)
            inlier[np.flatnonzero(ok)[keep]] = True
            trace["rounds"].append(dict(range=rng, pred=np.stack([x0, y0], 1), found=np.stack([x1, y1], 1),
                                        ok=ok.copy(), inlier=inlier))
        if not order:
            break
        mp.add_poly(order, coef)
        if use_resid:
            fitted = poly_feats(pc[ok, 0], pc[ok, 1], order) @ coef
            R = np.zeros((121, 2))
            idx = np.flatnonzero(ok)[keep]
            R[idx] = (d[ok] - fitted)[keep]
            mp.resid = R.reshape(11, 11, 2)

    # 남은 정렬 오차 측정
    off, ok, _ = refine_pins(Y, mp, ctx, 0.3, 0.05, ideal)
    pin_rms = float(np.sqrt((off[ok] ** 2).sum(1).mean())) if ok.any() else float("nan")
    return mp, dict(quad=quad, rot=rot, mirror=mirror, orient_ratio=orient_ratio,
                    cell_px=float(cell_px), pins_ok=int(ok.sum()), pin_rms=pin_rms, pin_ok_mask=ok)


# ---------------------------------------------------------------- 색 판정

def sample_cells(rgb, mp):
    g = np.arange(S) + 0.5
    U, V = np.meshgrid(g, g)
    su, sv = [a.ravel() for a in np.meshgrid(SUB, SUB)]
    x, y = mp(U[..., None] + su, V[..., None] + sv)
    return sample(rgb, x, y).mean(2)  # (S, S, 3)


def anchor_centroids(X):
    C = np.zeros((4, 4, 3))
    for a, (ox, oy) in enumerate(ANCHOR_ORIGINS):
        for qy in (0, 1):
            for qx in (0, 1):
                ys = slice(oy + 4 * qy + 1, oy + 4 * qy + 3)
                xs = slice(ox + 4 * qx + 1, ox + 4 * qx + 3)
                C[a, ANCHOR_PATTERN[qy][qx]] = X[ys, xs].reshape(-1, 3).mean(0)
    return C


def anchor_field(C):
    """앵커 4개의 중심값을 프레임 전체로 쌍선형 보간 (S, S, 4, 3)."""
    w = np.clip((np.arange(S) + 0.5 - 7) / 178, 0, 1)
    wx = w[None, :, None, None]
    wy = w[:, None, None, None]
    return ((1 - wx) * (1 - wy) * C[0] + wx * (1 - wy) * C[1]
            + (1 - wx) * wy * C[2] + wx * wy * C[3])


def _box3(a):
    p = np.pad(a, [(1, 1), (1, 1)] + [(0, 0)] * (a.ndim - 2))
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3))


def local_field(X, labels, usable, fallback):
    """16×16 블록(주변 3×3 블록 포함)마다 색별 평균 → 셀 단위로 보간 (S, S, 4, 3)."""
    B = S // 16
    F = np.zeros((S, S, 4, 3))
    for c in range(4):
        m = ((labels == c) & usable).astype(float)
        sums = _box3((X * m[..., None]).reshape(B, 16, B, 16, 3).sum((1, 3)))
        cnt = _box3(m.reshape(B, 16, B, 16).sum((1, 3)))
        mean = np.where((cnt >= 8)[..., None], sums / np.maximum(cnt, 1)[..., None], fallback[c])
        F[:, :, c] = cv2.resize(mean.astype(np.float32), (S, S), interpolation=cv2.INTER_LINEAR)
    return F


def classify_mahal(X, F, labels, usable):
    """공통 공분산(LDA)으로 거리 계산."""
    ys, xs = np.nonzero(usable)
    diff = X[ys, xs] - F[ys, xs, labels[ys, xs]]
    cov = diff.T @ diff / max(len(diff), 1) + 1e-6 * np.eye(3)
    inv = np.linalg.inv(cov)
    D = X[:, :, None, :] - F
    return np.einsum("...i,ij,...j->...", D, inv, D).argmin(-1)


def classify(X, ctx):
    """(앵커 기준 최근접, 판정 지향 국소 보정) 두 가지 판정 결과."""
    L = ctx.layout
    C = anchor_centroids(X)
    pred_anchor = ((X[:, :, None, :] - anchor_field(C)) ** 2).sum(-1).argmin(-1)

    usable = L.region != BORDER
    fallback = C.mean(0)
    lab = pred_anchor.copy()
    lab[ctx.fixed_mask] = L.fixed[ctx.fixed_mask]
    for _ in range(2):
        F = local_field(X, lab, usable, fallback)
        pred = classify_mahal(X, F, lab, usable)
        lab = pred.copy()
        lab[ctx.fixed_mask] = L.fixed[ctx.fixed_mask]
    return pred_anchor, pred, fallback


def classify_oracle(X, truth, fallback):
    """정답 라벨로 국소 중심·공분산을 잡은 판정 (분류기 한계 확인용)."""
    usable = np.ones((S, S), bool)
    F = local_field(X, truth, usable, fallback)
    return classify_mahal(X, F, truth, usable)


# ---------------------------------------------------------------- 채점

def rs_whatif(err_raster):
    """셀 오류를 RS 코드워드 바이트 오류 수로 환산 (배치 방식 3가지)."""
    e = err_raster[:8415 * 4].reshape(8415, 4).any(1)
    i = np.arange(8415)
    band = i // 2805
    return {
        "seq": e.reshape(33, 255).sum(1),                                     # 현행: 순차
        "ilv": np.bincount(i % 33, weights=e, minlength=33),                  # 전역 i mod 33
        "band": np.bincount(band * 11 + (i - band * 2805) % 11, weights=e, minlength=33),  # 3띠 × i mod 11
    }


def torn_check(pred, ctx, b):
    if ctx.n < 3:
        return False
    dm = ctx.data_mask
    cands = [b, (b - 1) % ctx.n, (b + 1) % ctx.n]
    errs = [(pred != ctx.expected[c]) & dm for c in cands]
    for axis in (1, 0):  # 행 방향 띠, 열 방향 띠
        cnt = dm.sum(axis).reshape(24, 8).sum(1)
        eb = [e.sum(axis).reshape(24, 8).sum(1) / np.maximum(cnt, 1) for e in errs]
        if ((eb[0] > 0.15) & (np.minimum(eb[1], eb[2]) < 0.5 * eb[0])).any():
            return True
    return False


def analyze_image(bgr, ctx, oracle=True):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    loc = locate(rgb, ctx)
    if loc is None:
        return {"status": "nodetect"}
    mp, geo = loc
    X = sample_cells(rgb, mp)
    pred_anchor, pred_dd, fallback = classify(X, ctx)

    pd = pred_dd.reshape(-1)[ctx.dflat]
    errs = (ctx.exp_data != pd[None, :]).sum(1)
    b = int(errs.argmin())
    nd = ctx.exp_data.shape[1]
    truth = ctx.expected[b]
    dm = ctx.data_mask

    res = dict(geo)
    res.update(best=b, ser_dd=errs[b] / nd,
               ser_2nd=float(np.partition(errs, 1)[1] / nd) if ctx.n > 1 else float("nan"),
               ser_anchor=float(((pred_anchor != truth) & dm).sum() / nd))
    if oracle:
        pred_or = classify_oracle(X, truth, fallback)
        res["ser_oracle"] = float(((pred_or != truth) & dm).sum() / nd)

    Y = X @ LUMA
    L = ctx.layout
    pf, ps = L.pair_first, L.pair_second
    res["border_agree"] = float(((Y.flat[pf] > Y.flat[ps]) == (truth.flat[pf] == WHITE)).mean())

    res["torn"] = torn_check(pred_dd, ctx, b)
    if res["torn"]:
        res["status"] = "torn"
    elif res["ser_dd"] > 0.3:
        res["status"] = "misaligned"
    else:
        res["status"] = "clean"

    err = (pred_dd != truth) & dm
    res["err_map"] = err
    res["rs"] = rs_whatif(err.reshape(-1)[ctx.dflat])
    conf = np.zeros((4, 4), int)
    np.add.at(conf, (truth[dm], pred_dd[dm]), 1)
    res["confusion"] = conf
    res["_mp"], res["_X"], res["_pred"] = mp, X, pred_dd
    return res


# ---------------------------------------------------------------- 디버그 이미지

def rectify(rgb_u8, mp, scale=6):
    g = (np.arange(S * scale) + 0.5) / scale
    U, V = np.meshgrid(g, g)
    x, y = mp(U, V)
    return sample(rgb_u8, x, y)


def overlay(bgr, mp, pin_ok, ctx):
    img = bgr.copy()
    t = max(2, round(img.shape[1] / 600))
    for k in list(range(0, S + 1, 16)) + [S]:
        s = np.linspace(0, S, 97)
        for u, v in ((np.full_like(s, k), s), (s, np.full_like(s, k))):
            x, y = mp(u, v)
            cv2.polylines(img, [np.stack([x, y], 1).astype(np.int32)], False, (255, 0, 255), t)
    x, y = mp(ctx.pin_uv[:, 0], ctx.pin_uv[:, 1])
    for xi, yi, ok in zip(x, y, pin_ok):
        cv2.circle(img, (int(xi), int(yi)), 3 * t, (0, 255, 255) if ok else (0, 0, 255), t)
    return img
