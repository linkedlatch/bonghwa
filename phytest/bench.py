#!/usr/bin/env python3
"""수신 알고리즘 v1(기준선) vs v2(현재) 비교. 같은 캡처를 두 방식으로 분석한다.

    python3 bench.py captures/IMG_0001.MOV captures/IMG_0002.MOV ... --out bench_result
"""
import argparse
import json
import multiprocessing as mproc
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # numpy/BLAS 스레드도 1개로 (병렬은 프로세스로)
import cv2
import numpy as np

import vdsp_rx
import vdsp_rx_v1

_CTX = None


def _init(args):
    global _CTX
    cv2.setNumThreads(1)  # 프로세스 병렬과 OpenCV 내부 스레드가 겹치지 않게
    _CTX = (vdsp_rx_v1.Context(*args), vdsp_rx.Context(*args))


def _work(item):
    src, fno, bgr = item
    c1, c2 = _CTX
    out = {"src": src, "frame": fno}
    r1 = vdsp_rx_v1.analyze_image(bgr, c1, oracle=False)
    out["v1"] = {"status": r1["status"]}
    if r1["status"] != "nodetect":
        out["v1"].update(best=r1["best"], ser=float(r1["ser_dd"]),
                         dec={k: bool((v <= 12).all()) for k, v in r1["rs"].items()},
                         cw={k: (v <= 12).tolist() for k, v in r1["rs"].items()})
    r2 = vdsp_rx.analyze_image(bgr, c2, oracle=False)
    out["v2"] = {"status": r2["status"]}
    if r2["status"] != "nodetect":
        out["v2"].update(best=r2["best"], ser=float(r2["ser_dd"]), visible=r2["visible_frac"],
                         ser_v1geo=r2["ser_v1"], ser_em=r2["ser_em"], ser_local=r2["ser_local"],
                         dec={k: bool(v.all()) for k, v in r2["rs_ok"].items()},
                         gmd={k: bool(v.all()) for k, v in r2["rs_gmd"].items()},
                         cw={k: v.tolist() for k, v in r2["rs_gmd"].items()})
    return out


def frames(path, step):
    cap = cv2.VideoCapture(path)
    i = 0
    while cap.grab():
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok:
                yield os.path.basename(path), i, img
        i += 1


def unwrap(seq, n):
    out, prev = [], None
    for b in seq:
        if b is None:
            out.append(None)
            continue
        if prev is None:
            u = b
        else:
            d = (b - prev[0]) % n
            u = prev[1] + (d if d <= n // 2 else d - n)
        out.append(u)
        prev = (b, u)
    return out


def summarize(rows, n):
    """영상 하나의 v1/v2 지표."""
    res = {}
    # 표시 프레임 번호: v2가 식별한 번호를 기준으로 펼친다 (정렬 실패는 제외)
    ref = [r["v2"]["best"] if r["v2"]["status"] in ("clean", "torn") else None for r in rows]
    uw = unwrap(ref, n)
    valid = [u for u in uw if u is not None]
    span = max(valid) - min(valid) + 1 if valid else 0
    for v in ("v1", "v2"):
        rs = [r[v] for r in rows]
        clean = [x for x in rs if x["status"] == "clean"]
        sers = np.array([x["ser"] for x in clean]) if clean else np.array([np.nan])
        d = {"captures": len(rs), "clean": len(clean) / len(rs),
             "ser_median": float(np.median(sers)), "ser_mean": float(np.mean(sers))}
        for key in ("seq", "ilv"):
            field = "gmd" if v == "v2" else "dec"
            ok = [x.get(field, {}).get(key, False) and x["status"] == "clean" for x in rs]
            d[f"cap_{key}"] = float(np.mean(ok))
            got = {u for u, o in zip(uw, ok) if o and u is not None}
            d[f"disp_{key}"] = len(got) / span if span else 0.0
            # 캡처 결합: 같은 표시 프레임의 정상 캡처에서 코드워드별로 성공한 것을 모은다
            groups = {}
            for u, x in zip(uw, rs):
                if u is not None and x["status"] == "clean" and "cw" in x:
                    groups.setdefault(u, []).append(np.array(x["cw"][key]))
            comb = sum(np.logical_or.reduce(g).all() for g in groups.values())
            d[f"comb_{key}"] = comb / span if span else 0.0
        if v == "v2":
            d["v2_dec_ilv_hard"] = float(np.mean([x.get("dec", {}).get("ilv", False) and x["status"] == "clean"
                                                  for x in rs]))
            for k in ("ser_v1geo", "ser_em", "ser_local", "ser"):
                a = [x[k] for x in clean]
                d[f"abl_{k}"] = float(np.median(a)) if a else float("nan")
        res[v] = d
    res["span"] = span
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="bench_result")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ctx_args = (1, args.n, "checker", "bin", "rgbw")

    allres = {}
    t0 = time.time()
    with mproc.Pool(args.jobs, _init, (ctx_args,)) as pool:
        for path in args.inputs:
            rows = []
            batch = []
            for item in frames(path, args.step):
                batch.append(item)
                if len(batch) >= args.jobs * 2:
                    rows += pool.map(_work, batch)
                    batch = []
            if batch:
                rows += pool.map(_work, batch)
            name = os.path.basename(path)
            with open(os.path.join(args.out, f"{name}.json"), "w") as f:
                json.dump(rows, f)
            allres[name] = summarize(rows, args.n)
            print(f"{name}: {len(rows)} 캡처 ({time.time() - t0:.0f}s)", file=sys.stderr)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(allres, f, indent=2, ensure_ascii=False)

    L = []
    p = L.append
    p("# 수신 알고리즘 v1 → v2 비교 (전역 인터리빙 = i mod 33, 현행 = 순차)\n")
    p("| 영상 | 방식 | 정상 캡처 | 셀 오류율 중앙/평균 | 캡처 복호 (현행/인터리빙) | 표시 프레임 복호 (현행/인터리빙) | + 캡처 결합 (인터리빙) |")
    p("|---|---|---|---|---|---|---|")
    for name, r in allres.items():
        for v in ("v1", "v2"):
            d = r[v]
            p(f"| {name} | {v} | {d['clean']:.1%} | {d['ser_median']:.2%} / {d['ser_mean']:.2%} | "
              f"{d['cap_seq']:.1%} / {d['cap_ilv']:.1%} | {d['disp_seq']:.1%} / {d['disp_ilv']:.1%} | "
              f"{d['comb_ilv']:.1%} |")
    p("\n## v2 단계별 셀 오류율 (정상 캡처 중앙값)\n")
    p("| 영상 | v1 판정 (v2 기하) | + 기지 셀 EM | + 블록 이득·오프셋 | + 3×3 등화 | 인터리빙 복호: 경판정 → GMD |")
    p("|---|---|---|---|---|---|")
    for name, r in allres.items():
        d = r["v2"]
        p(f"| {name} | {d['abl_ser_v1geo']:.2%} | {d['abl_ser_em']:.2%} | {d['abl_ser_local']:.2%} | "
          f"{d['abl_ser']:.2%} | {d['v2_dec_ilv_hard']:.1%} → {d['cap_ilv']:.1%} |")
    text = "\n".join(L)
    print(text)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
