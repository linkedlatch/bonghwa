#!/usr/bin/env python3
"""VDSP 물리 계층 테스트: 촬영한 동영상/사진에서 셀 오인식률을 측정한다.

    python3 analyze.py VIDEO.mp4 [더 많은 파일 ...] --out result/

송신기(tx.html)와 같은 --seed / --n / --pin / --counter를 써야 한다.
"""
import argparse
import csv
import json
import multiprocessing as mproc
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # numpy/BLAS 스레드도 1개로 (병렬은 프로세스로)
import cv2
import numpy as np

from vdsp_layout import S, DATA, render
from vdsp_rx import Context, analyze_image, overlay, rectify

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".heic"}
LAYOUTS = (("seq", "순차 (현행 spec)"), ("ilv", "전역 i mod 33"), ("band", "3띠 × i mod 11"))


def iter_inputs(paths, step, start, end, limit):
    count = 0
    for path in paths:
        if os.path.splitext(path)[1].lower() in IMAGE_EXT:
            img = cv2.imread(path)
            if img is None:
                print(f"읽기 실패: {path}", file=sys.stderr)
                continue
            yield path, 0, 0.0, img
            count += 1
            if limit and count >= limit:
                return
            continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"영상 열기 실패: {path} (ffmpeg로 H.264 mp4 변환 후 재시도)", file=sys.stderr)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while cap.grab():
            t = i / fps
            if end and t > end:
                break
            if t >= start and (i % step) == 0:
                ok, img = cap.retrieve()
                if ok:
                    yield path, i, t, img
                    count += 1
                    if limit and count >= limit:
                        return
            i += 1
        cap.release()


_WORKER_CTX = None


def _init_worker(ctx_args):
    global _WORKER_CTX
    cv2.setNumThreads(1)  # 프로세스 병렬과 OpenCV 내부 스레드가 겹치지 않게
    _WORKER_CTX = Context(*ctx_args)


def _work(item):
    bgr, oracle = item
    return analyze_image(bgr, _WORKER_CTX, oracle=oracle)


def analyzed(items, ctx, ctx_args, oracle, jobs):
    """(src, fno, t, bgr)를 받아 (src, fno, t, bgr, res)를 순서대로 내보낸다."""
    if jobs <= 1:
        for src, fno, t, bgr in items:
            yield src, fno, t, bgr, analyze_image(bgr, ctx, oracle=oracle)
        return
    with mproc.Pool(jobs, _init_worker, (ctx_args,)) as pool:
        batch = []
        for it in items:
            batch.append(it)
            if len(batch) >= jobs * 2:
                for b, res in zip(batch, pool.map(_work, [(b[3], oracle) for b in batch])):
                    yield (*b, res)
                batch = []
        if batch:
            for b, res in zip(batch, pool.map(_work, [(b[3], oracle) for b in batch])):
                yield (*b, res)


def save_debug(outdir, tag, bgr, res, ctx):
    os.makedirs(outdir, exist_ok=True)
    mp = res["_mp"]
    ov = overlay(bgr, mp, res["pin_ok_mask"], ctx)
    scale = 1600 / max(ov.shape[:2])
    if scale < 1:
        ov = cv2.resize(ov, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(outdir, f"{tag}_overlay.jpg"), ov)

    k = 5
    rect = rectify(bgr, mp, k)
    truth = render(ctx.expected[res["best"]], k, 0, ctx.palette)[:, :, ::-1]
    err = np.repeat(np.repeat(res["err_map"], k, 0), k, 1)
    errimg = rect.copy()
    errimg[err] = (255, 0, 255)
    errimg[~err] = (errimg[~err] * 0.35).astype(np.uint8)
    cv2.imwrite(os.path.join(outdir, f"{tag}_rect_truth_err.png"), np.hstack([rect, truth, errimg]))


def unwrap(seq, n):
    out, prev = [], None
    for b in seq:
        if prev is None:
            u = b
        else:
            d = (b - prev[0]) % n
            u = prev[1] + (d if d <= n // 2 else d - n)
        out.append(u)
        prev = (b, u)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="동영상 또는 사진 파일")
    ap.add_argument("--out", default="phy_result", help="결과 폴더 (기본 phy_result)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--pin", default="checker", choices=["checker", "cross"])
    ap.add_argument("--counter", default="bin", choices=["bin", "gray"])
    ap.add_argument("--palette", default="rgbw", choices=["rgbw", "cmyg"])
    ap.add_argument("--step", type=int, default=1, help="k번째 영상 프레임마다 처리")
    ap.add_argument("--start", type=float, default=0, help="시작 시각(초)")
    ap.add_argument("--end", type=float, default=0, help="끝 시각(초), 0이면 끝까지")
    ap.add_argument("--max", type=int, default=0, help="최대 처리 캡처 수")
    ap.add_argument("--debug", type=int, default=3, help="디버그 이미지 저장 개수 (처음/최악/찢김 각각)")
    ap.add_argument("--no-oracle", action="store_true", help="정답 기반 판정 생략")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="병렬 프로세스 수 (기본: CPU 수 - 1)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dbg = os.path.join(args.out, "debug")
    ctx_args = (args.seed, args.n, args.pin, args.counter, args.palette)
    ctx = Context(*ctx_args)

    rows = []
    heat_err = np.zeros((S, S))
    heat_n = 0
    conf = np.zeros((4, 4), int)
    worst = []          # (ser, tag, bgr, res)
    saved = {"clean": 0, "torn": 0, "misaligned": 0}
    groups = []         # [(src, best, [cw_ok dict...])]
    t_start = time.time()

    items = iter_inputs(args.inputs, args.step, args.start, args.end, args.max)
    for n_done, (src, fno, t, bgr, res) in enumerate(
            analyzed(items, ctx, ctx_args, not args.no_oracle, args.jobs), 1):
        st = res["status"]
        row = dict(src=os.path.basename(src), frame=fno, t=round(t, 3), status=st)
        if st != "nodetect":
            row.update(best=res["best"], rot=res["rot"] * 90, mirror=res["mirror"],
                       orient_ratio=round(res["orient_ratio"], 2), cell_px=round(res["cell_px"], 2),
                       pins_ok=res["pins_ok"], pin_rms_cell=round(res["pin_rms"], 3),
                       visible=round(res["visible_frac"], 4),
                       ser_anchor=res["ser_anchor"], ser_v1=res["ser_v1"], ser_em=res["ser_em"],
                       ser_local=res["ser_local"], ser_dd=float(res["ser_dd"]),
                       ser_oracle=res.get("ser_oracle", ""), ser_2nd=res["ser_2nd"],
                       border_agree=round(res["border_agree"], 4), torn=int(res["torn"]))
            for key, _ in LAYOUTS:
                row[f"maxcw_{key}"] = int(res["rs"][key].max())
                row[f"dec_{key}"] = int(res["rs_ok"][key].all())
                row[f"gmd_{key}"] = int(res["rs_gmd"][key].all())
            tag = f"{os.path.splitext(row['src'])[0]}_{fno:06d}"
            if st == "clean":
                heat_err += res["err_map"]
                heat_n += 1
                conf += res["confusion"]
                if not groups or groups[-1][0] != src or groups[-1][1] != res["best"]:
                    groups.append([src, res["best"], []])
                groups[-1][2].append({k: res["rs_gmd"][k] for k, _ in LAYOUTS})
            if saved.get(st, args.debug) < args.debug:
                save_debug(dbg, f"{st}_{tag}", bgr, res, ctx)
                saved[st] += 1
            if st in ("clean", "torn"):
                worst.append((float(res["ser_dd"]), tag, bgr, res))
                worst.sort(key=lambda w: -w[0])
                del worst[args.debug:]
        rows.append(row)
        if n_done % 20 == 0:
            print(f"  {n_done} 캡처 처리 ({time.time() - t_start:.0f}s)", file=sys.stderr)

    for ser, tag, bgr, res in worst:
        save_debug(dbg, f"worst_{tag}", bgr, res, ctx)

    # ---------------------------------------------------------- CSV
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(args.out, "captures.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------- 요약
    lines = []
    p = lines.append
    total = len(rows)
    by = {s: [r for r in rows if r["status"] == s] for s in ("clean", "torn", "misaligned", "nodetect")}
    p(f"# VDSP PHY 테스트 결과  (seed {args.seed}, n {args.n}, pin {args.pin}, counter {args.counter}, "
      f"palette {args.palette})")
    p("")
    p(f"처리한 캡처: {total}")
    for s, name in (("clean", "정상"), ("torn", "찢김"), ("misaligned", "정렬 실패(SER>30%)"),
                    ("nodetect", "프레임 검출 실패")):
        p(f"  {name:<18} {len(by[s]):6d}  ({len(by[s]) / max(total, 1):.1%})")

    clean = by["clean"]
    summary = {"captures": total, **{s: len(v) for s, v in by.items()}}
    if clean:
        def stats(key):
            a = np.array([r[key] for r in clean if r.get(key, "") != ""], float)
            return a
        p("")
        p("## 셀 오인식률 (정상 캡처 기준, 화면 안 데이터 셀)")
        p(f"  {'판정 방식':<30}{'중앙값':>10}{'평균':>10}{'p90':>10}{'최대':>10}")
        for key, name in (("ser_anchor", "앵커 기준 최근접"),
                          ("ser_v1", "v1: 앵커 초기값 + 국소 보정"),
                          ("ser_em", "v2-1: 기지 셀 + 전역 EM"),
                          ("ser_local", "v2-2: + 블록별 이득·오프셋"),
                          ("ser_dd", "v2-3: + 3×3 이웃 등화 (기본)"),
                          ("ser_oracle", "정답 기반 v1 (진단용)")):
            a = stats(key)
            if len(a):
                p(f"  {name:<28}{np.median(a):>10.4%}{a.mean():>10.4%}"
                  f"{np.percentile(a, 90):>10.4%}{a.max():>10.4%}")
                summary[key] = {"median": float(np.median(a)), "mean": float(a.mean()),
                                "p90": float(np.percentile(a, 90))}
        cp = stats("cell_px")
        pr = stats("pin_rms_cell")
        po = stats("pins_ok")
        p(f"  셀당 카메라 px: 중앙값 {np.median(cp):.2f}  (최소 {cp.min():.2f})")
        p(f"  정렬 핀: 검출 {np.median(po):.0f}/121개 (중앙값), 잔여 정렬 오차 RMS {np.nanmedian(pr):.3f}셀")
        ba = stats("border_agree")
        p(f"  테두리 카운터 쌍 판독 일치율: 중앙값 {np.median(ba):.2%}")

        p("")
        p("## 색 혼동 행렬 (정상 캡처 합계, 행 = 실제, 열 = 판정)")
        p("        " + "".join(f"{c:>12}" for c in ctx.color_names) + "     오류율")
        for i, c in enumerate(ctx.color_names):
            tot = conf[i].sum()
            p(f"  {c:<6}" + "".join(f"{conf[i, j]:>12d}" for j in range(4))
              + f"   {1 - conf[i, i] / max(tot, 1):8.4%}")
        summary["confusion"] = conf.tolist()

        p("")
        p("## RS(255,231) 복호 가능 여부 추정 (t=12, 셀 오류 → 바이트 오류, 화면 밖 셀 = 소거)")
        p(f"  {'배치 방식':<22}{'캡처(경판정)':>12}{'캡처(GMD)':>11}{'표시 프레임':>12}{'+ 캡처 결합':>12}")
        summary["decode"] = {}
        for key, name in LAYOUTS:
            per_cap = np.mean([r[f"dec_{key}"] for r in clean])
            per_gmd = np.mean([r[f"gmd_{key}"] for r in clean])
            single = np.mean([any(c[key].all() for c in g[2]) for g in groups])
            comb = np.mean([np.logical_or.reduce([c[key] for c in g[2]]).all() for g in groups])
            p(f"  {name:<22}{per_cap:>12.1%}{per_gmd:>11.1%}{single:>12.1%}{comb:>12.1%}")
            summary["decode"][key] = {"capture": float(per_cap), "capture_gmd": float(per_gmd),
                                      "display_frame": float(single), "combined": float(comb)}
        p("  (경판정: 2·오류+소거 ≤ 24. GMD: 신뢰도 낮은 바이트를 추가로 소거해 재시도.")
        p("   표시 프레임·캡처 결합은 GMD 기준. 캡처 결합 = 코드워드마다 복호된 캡처를 골라 조합)")

        # 재생 모드: 표시 프레임 확보율
        p("")
        p("## 표시 프레임 확보 (재생 모드)")
        any_anim = False
        for src in dict.fromkeys(r["src"] for r in rows):
            rs = [r for r in rows if r["src"] == src and r["status"] in ("clean", "torn")]
            if len(rs) < 2:
                continue
            u = unwrap([r["best"] for r in rs], args.n)
            span = max(u) - min(u) + 1
            if span <= 1:
                continue
            any_anim = True
            got = {ui for ui, r in zip(u, rs) if r["status"] == "clean"}
            missing = sorted(set(range(min(u), max(u) + 1)) - got)
            longest, run, last = 0, 0, None
            for m in missing:
                run = run + 1 if last is not None and m == last + 1 else 1
                longest, last = max(longest, run), m
            dur = rs[-1]["t"] - rs[0]["t"]
            p(f"  {src}: 표시된 프레임 {span}개 중 정상 캡처 확보 {len(got)}개 ({len(got) / span:.1%}), "
              f"최장 연속 누락 {longest}개, 측정 송출 속도 {span / dur if dur else 0:.2f} fps")
        if not any_anim:
            p("  (정지 모드이거나 프레임 변화 없음)")

        # 오류 분포 이미지
        rate = heat_err / heat_n
        vis = (np.clip(rate / max(rate.max(), 1e-9), 0, 1) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        vis[ctx.layout.region != DATA] = (60, 60, 60)
        vis = cv2.resize(vis, (S * 4, S * 4), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.out, "error_heatmap.png"), vis)
        p("")
        p(f"오류 분포: error_heatmap.png (셀별 오류율, 최대 {rate.max():.2%}, 회색 = 데이터 아닌 셀)")
    else:
        p("")
        p("정상 캡처가 없습니다. debug/ 폴더의 overlay 이미지로 검출·정렬 상태를 확인하세요.")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
