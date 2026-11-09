#!/usr/bin/env python3
"""캡처마다 이웃 표시 프레임(앞/뒤)이 얼마나 섞였는지 잰다 (찢김 검출기보다 민감한 측정).

프레임 행을 8행씩 24개 띠로 나누고, 띠마다 '판정값이 기준 프레임과는 다르고 앞 또는 뒤 프레임과 같은'
셀의 비율(이웃 설명률)을 구한다. 이웃 프레임 비율이 f인 띠에서 이 값은 약 0.75·f이다.
"""
import argparse
import multiprocessing as mproc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
import numpy as np

from vdsp_layout import S
from vdsp_rx import Context, locate, sample_cells, classify_v2

_CTX = None


def _init():
    global _CTX
    cv2.setNumThreads(1)
    _CTX = Context()


def _work(item):
    fno, bgr = item
    ctx = _CTX
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    loc = locate(rgb, ctx)
    if loc is None:
        return None
    mp, geo = loc
    vis = geo["visible"]
    pred = classify_v2(sample_cells(rgb, mp), ctx, vis)["neigh"]
    dm = ctx.data_mask & vis
    pd = pred.reshape(-1)[ctx.dflat]
    errs = ((ctx.exp_data != pd) & vis.reshape(-1)[ctx.dflat]).sum(1)
    b = int(errs.argmin())
    n = ctx.n
    E = ctx.expected
    wrong = (pred != E[b]) & dm
    nb = wrong & ((pred == E[(b - 1) % n]) | (pred == E[(b + 1) % n]))
    cnt = dm.sum(1).reshape(24, 8).sum(1)
    band_nb = nb.sum(1).reshape(24, 8).sum(1) / np.maximum(cnt, 1)
    band_err = wrong.sum(1).reshape(24, 8).sum(1) / np.maximum(cnt, 1)
    return dict(frame=fno, ser=float(wrong.sum() / max(dm.sum(), 1)), band_nb=band_nb.tolist(),
                band_err=band_err.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    with mproc.Pool(args.jobs, _init) as pool:
        for path in args.inputs:
            cap = cv2.VideoCapture(path)
            items, i = [], 0
            res = []
            while cap.grab():
                if i % args.step == 0:
                    items.append((i, cap.retrieve()[1]))
                    if len(items) >= args.jobs * 2:
                        res += pool.map(_work, items)
                        items = []
                i += 1
            res += pool.map(_work, items)
            res = [r for r in res if r]
            mx = np.array([max(r["band_nb"]) for r in res])
            nbands = np.array([sum(x > 0.02 for x in r["band_nb"]) for r in res])
            print(f"\n## {os.path.basename(path)}  ({len(res)} 캡처)")
            print("  띠 중 최대 이웃 설명률 분포 (섞인 비율 f ≈ 값/0.75):")
            for lo, hi, name in ((0, 0.005, "섞임 없음 (<0.5%)"), (0.005, 0.02, "흔적 (0.5~2%)"),
                                 (0.02, 0.15, "부분 섞임 (2~15%)"), (0.15, 1.01, "찢김 (≥15%)")):
                m = (mx >= lo) & (mx < hi)
                print(f"    {name:<20} {m.mean():6.1%}   이 캡처들의 SER 중앙값 "
                      f"{np.median([r['ser'] for r, k in zip(res, m) if k]) if m.any() else float('nan'):.3%}")
            sel = mx >= 0.02
            if sel.any():
                print(f"  섞인 캡처에서 섞인 띠 수(8행 단위): 중앙값 {np.median(nbands[sel]):.0f} / 24")
                prof = np.mean([r["band_nb"] for r, k in zip(res, sel) if k], 0)
                print("  섞인 캡처 평균 띠별 이웃 설명률 (위→아래): " + " ".join(f"{v:.2f}" for v in prof))


if __name__ == "__main__":
    main()
