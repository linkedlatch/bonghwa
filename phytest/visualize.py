#!/usr/bin/env python3
"""검출 과정 시각화.

    python3 visualize.py VIDEO --frame 100 --out viz/     # 단계별 이미지 (steps_000100.png)
    python3 visualize.py VIDEO --video --out viz/         # 프레임만 펴서 담은 영상 (rectified.webm)

영상은 [원본 + 검출 격자 | 보정된 프레임 | 오류 위치] 3칸으로 구성된다.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from vdsp_layout import S, ANCHOR_ORIGINS, ANCHOR_PATTERN, DATA, palette_rgb
from vdsp_rx import (Context, Mapping, locate, sample_cells, classify, torn_check, rectify, overlay,
                     sample, _AQ_PTS)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def caption(img, text, sub=""):
    bar = np.full((56, img.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (10, 24), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (10, 47), FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit(img, size):
    """비율을 유지해 size×size 안에 넣고 나머지는 검정."""
    h, w = img.shape[:2]
    k = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))),
                   interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_NEAREST)
    out = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def crop_box(quad, shape, margin=0.08):
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    m = margin * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    x1, y1 = int(min(shape[1], x1 + m)), int(min(shape[0], y1 + m))
    return x0, y0, x1, y1


def panel_detect(bgr, quad, size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    V = (rgb.max(axis=2) * 255).astype(np.uint8)
    _, th = cv2.threshold(cv2.GaussianBlur(V, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.cvtColor(th // 2 + 40, cv2.COLOR_GRAY2BGR)
    t = max(2, bgr.shape[1] // 400)
    cv2.polylines(img, [quad.astype(np.int32)], True, (0, 255, 0), t)
    for i, (x, y) in enumerate(quad):
        cv2.circle(img, (int(x), int(y)), 4 * t, (0, 200, 255), -1)
        cv2.putText(img, str(i), (int(x) + 6 * t, int(y) - 6 * t), FONT, 0.6 * t, (0, 200, 255), t)
    return fit(img, size)


def panel_anchors(bgr, H, size, names):
    """앵커 4개 주변을 호모그래피로 펴서 확대. 표본 위치와 기대 색 표시."""
    mp = Mapping(H)
    cell = size // 2 // 12
    tiles = []
    for ox, oy in ANCHOR_ORIGINS:
        g = (np.arange(12 * cell) + 0.5) / cell
        U, V = np.meshgrid(ox - 2 + g, oy - 2 + g)
        x, y = mp(U, V)
        tile = sample(bgr, x, y).copy()
        for qx in (0, 1):
            for qy in (0, 1):
                col = ANCHOR_PATTERN[qy][qx]
                cx, cy = (2 + 4 * qx + 2) * cell, (2 + 4 * qy + 2) * cell
                for du in np.linspace(-1, 1, 5):
                    for dv in np.linspace(-1, 1, 5):
                        cv2.circle(tile, (int(cx + du * cell), int(cy + dv * cell)), 2, (0, 0, 0), -1)
                cv2.putText(tile, names[col], (cx - 9, cy + 30), FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(tile, names[col], (cx - 9, cy + 30), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(tile, (2 * cell, 2 * cell), (10 * cell, 10 * cell), (0, 255, 255), 2)
        tiles.append(tile)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    return fit(grid, size)


def panel_pins(bgr, trace, mp, box, size, magnify=8):
    x0, y0, x1, y1 = box
    img = (bgr * 0.6).astype(np.uint8)
    # 최종 매핑의 16셀 격자
    for k in range(0, S + 1, 16):
        s = np.linspace(0, S, 97)
        for u, v in ((np.full_like(s, k), s), (s, np.full_like(s, k))):
            x, y = mp(u, v)
            cv2.polylines(img, [np.stack([x, y], 1).astype(np.int32)], False, (255, 255, 0), 1)
    r = trace["rounds"][0]
    t = max(1, round((x1 - x0) / 500))
    for p, f, ok, inl in zip(r["pred"], r["found"], r["ok"], r["inlier"]):
        tip = p + (f - p) * magnify
        if not ok:
            cv2.drawMarker(img, tuple(p.astype(int)), (128, 128, 128), cv2.MARKER_TILTED_CROSS, 8 * t, t)
            continue
        color = (0, 255, 0) if inl else (0, 128, 255)
        cv2.circle(img, tuple(p.astype(int)), 3 * t, (0, 0, 255), -1)
        cv2.arrowedLine(img, tuple(p.astype(int)), tuple(tip.astype(int)), color, t, tipLength=0.3)
    return fit(img[y0:y1, x0:x1], size)


def panel_errors(rect, err, k):
    e = np.repeat(np.repeat(err, k, 0), k, 1)
    out = (rect * 0.35).astype(np.uint8)
    out[e] = (255, 0, 255)
    return out


def analyze(bgr, ctx, trace=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    loc = locate(rgb, ctx, trace)
    if loc is None:
        return None
    mp, geo = loc
    X = sample_cells(rgb, mp)
    _, pred, _ = classify(X, ctx)
    pd = pred.reshape(-1)[ctx.dflat]
    errs = (ctx.exp_data != pd).sum(1)
    b = int(errs.argmin())
    truth = ctx.expected[b]
    err = (pred != truth) & ctx.data_mask
    status = "torn" if torn_check(pred, ctx, b) else ("misaligned" if errs[b] / len(pd) > 0.3 else "clean")
    return dict(mp=mp, geo=geo, pred=pred, best=b, err=err, ser=errs[b] / len(pd), status=status, truth=truth)


def steps_image(bgr, ctx, size=720):
    trace = {}
    r = analyze(bgr, ctx, trace)
    if r is None:
        sys.exit("프레임 검출 실패")
    geo, mp = r["geo"], r["mp"]
    box = crop_box(geo["quad"], bgr.shape)
    k = 4
    rect = rectify(bgr, mp, k)
    truth = cv2.cvtColor(palette_rgb(ctx.palette)[r["truth"]].astype(np.uint8), cv2.COLOR_RGB2BGR)
    truth = np.repeat(np.repeat(truth, k, 0), k, 1)
    conf = np.zeros((4, 4), int)
    np.add.at(conf, (r["truth"][ctx.data_mask], r["pred"][ctx.data_mask]), 1)
    names = ctx.color_names
    worst = sorted(((conf[i, j], f"{names[i]}->{names[j]}") for i in range(4) for j in range(4)
                    if i != j), reverse=True)[:2]
    r0 = trace["rounds"][0]
    panels = [
        caption(fit(bgr, size), "1. Input capture", f"{bgr.shape[1]}x{bgr.shape[0]}"),
        caption(panel_detect(bgr, geo["quad"], size), "2. Frame detection",
                "Otsu threshold on max(R,G,B) -> largest quadrilateral"),
        caption(panel_anchors(bgr, trace["H"], size, ctx.color_names), "3. Orientation from anchors (corner homography)",
                f"rot {geo['rot'] * 90} deg, mirror {geo['mirror']}, best/2nd score {geo['orient_ratio']:.0f}x; "
                "offset = distortion before pin fix"),
        caption(panel_pins(bgr, trace, mp, box, size), "4. Alignment pins (round 1, arrows x8)",
                f"found {int(r0['ok'].sum())}/121, inliers {int(r0['inlier'].sum())} "
                f"(green=used, orange=outlier, x=not found); final residual {geo['pin_rms']:.3f} cell"),
        caption(fit(np.vstack([rect[:48 * k, :96 * k], truth[:48 * k, :96 * k]]), size),
                "5. Rectified (top) vs truth (bottom)",
                f"cells x 0-95, y 0-47; {geo['cell_px']:.2f} camera px/cell"),
        caption(fit(panel_errors(rect, r["err"], k), size), "6. Classification errors (magenta)",
                f"frame #{r['best']}  SER {r['ser']:.2%}  top confusions: "
                + ", ".join(f"{n} {c}" for c, n in worst)),
    ]
    return np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])]), r


def make_video(path, ctx, out, size=576, step=1, fps=None):
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = fps or src_fps / step
    writer = None
    i = n = 0
    k = 3
    stats = {"clean": 0, "torn": 0, "misaligned": 0, "nodetect": 0}
    while cap.grab():
        if i % step:
            i += 1
            continue
        bgr = cap.retrieve()[1]
        r = analyze(bgr, ctx)
        left = fit(bgr, size)
        if r is None:
            stats["nodetect"] += 1
            mid = right = np.zeros((size, size, 3), np.uint8)
            label = "no detection"
        else:
            stats[r["status"]] += 1
            left = fit(overlay(bgr, r["mp"], r["geo"]["pin_ok_mask"], ctx), size)
            rect = rectify(bgr, r["mp"], k)
            mid = fit(rect, size)
            right = fit(panel_errors(rect, r["err"], k), size)
            label = f"{r['status']}  display frame #{r['best']}  SER {r['ser']:.2%}  pins {r['geo']['pins_ok']}"
        frame = np.hstack([left, mid, right])
        bar = np.full((40, frame.shape[1], 3), 32, np.uint8)
        cv2.putText(bar, f"capture {i}  t={i / src_fps:6.2f}s  |  {label}", (10, 27), FONT, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.vstack([bar, frame])
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"VP90"), fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                sys.exit("VP9 webm 인코더를 열 수 없습니다")
        writer.write(frame)
        n += 1
        if n % 50 == 0:
            print(f"  {n} 프레임", file=sys.stderr)
        i += 1
    if writer:
        writer.release()
    return n, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", default="viz")
    ap.add_argument("--frame", type=int, default=None, help="단계별 이미지를 만들 영상 프레임 번호")
    ap.add_argument("--video", action="store_true", help="프레임만 펴서 담은 영상 생성")
    ap.add_argument("--step", type=int, default=1, help="영상: k프레임마다 사용")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--pin", default="checker", choices=["checker", "cross"])
    ap.add_argument("--counter", default="bin", choices=["bin", "gray"])
    ap.add_argument("--palette", default="rgbw", choices=["rgbw", "cmyg"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ctx = Context(args.seed, args.n, args.pin, args.counter, args.palette)
    base = os.path.splitext(os.path.basename(args.input))[0]

    if args.frame is not None:
        cap = cv2.VideoCapture(args.input)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, bgr = cap.read()
        if not ok:
            bgr = cv2.imread(args.input)
        img, r = steps_image(bgr, ctx)
        out = os.path.join(args.out, f"{base}_steps_{args.frame:06d}.png")
        cv2.imwrite(out, img)
        print(f"{out}  (frame #{r['best']}, SER {r['ser']:.2%}, {r['status']})")
    if args.video:
        out = os.path.join(args.out, f"{base}_rectified.webm")
        n, stats = make_video(args.input, ctx, out, step=args.step)
        print(f"{out}  ({n} frames, {os.path.getsize(out) / 1e6:.1f} MB)  {stats}")


if __name__ == "__main__":
    main()
