#!/usr/bin/env python3
"""분석기 자체 검증: JS/Python 프레임 일치 확인 + 합성 촬영 이미지로 검출·판정 시험."""
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

from vdsp_layout import Layout, gen_frame, render
from vdsp_rx import Context, analyze_image

HERE = os.path.dirname(os.path.abspath(__file__))


def _js_frame_fn():
    """JS genFrame을 부르는 함수. node 또는 mini-racer(pip install mini-racer)가 필요."""
    js = os.path.join(HERE, "vdsp_frame.js")
    node = shutil.which("node")
    if node:
        script = ("const V=require(process.argv[1]);const L=V.buildLayout(process.argv[2]);"
                  "process.stdout.write(Buffer.from(V.genFrame(L,+process.argv[3],"
                  "{seed:+process.argv[4],counter:process.argv[5]})));")

        def run(pin, idx, seed, counter):
            out = subprocess.run([node, "-e", script, js, pin, str(idx), str(seed), counter],
                                 capture_output=True, check=True).stdout
            return np.frombuffer(out, np.uint8).reshape(192, 192)
        return run
    try:
        from py_mini_racer import MiniRacer
    except ImportError:
        return None
    mr = MiniRacer()
    mr.eval("var self = this;" + open(js).read().replace("})(this);", "})(self);"))

    def run(pin, idx, seed, counter):
        out = mr.eval(f"Array.from(VDSP.genFrame(VDSP.buildLayout('{pin}'), {idx}, "
                      f"{{seed: {seed}, counter: '{counter}'}}))")
        return np.array(list(out), np.uint8).reshape(192, 192)
    return run


def check_js_parity():
    run = _js_frame_fn()
    if run is None:
        print("[건너뜀] node 또는 mini-racer가 없어 JS/Python 일치 검사를 생략합니다")
        return True
    ok = True
    for pin in ("checker", "cross"):
        L = Layout(pin)
        for counter in ("bin", "gray"):
            for idx, seed in ((0, 1), (5, 1), (63, 1), (7, 123456789), (2, 4294967295)):
                js_cells = run(pin, idx, seed, counter)
                py_cells = gen_frame(L, idx, seed, counter)
                if not np.array_equal(js_cells, py_cells):
                    ok = False
                    print(f"[실패] JS/Python 불일치 pin={pin} counter={counter} idx={idx} seed={seed}: "
                          f"{int((js_cells != py_cells).sum())}셀")
    print(f"[{'통과' if ok else '실패'}] JS/Python 프레임 일치")
    return ok


def synth(ctx, idx, rot=0, mirror=False, px=6, canvas=(1920, 1080), fill=0.8, persp=0.05,
          barrel=0.04, blur=0.8, noise=0.015, gain=(1.0, 0.92, 1.08), offset=0.03, jpeg=85,
          torn_with=None, seed=0):
    """송신 화면을 찍은 것처럼 만든 BGR 이미지."""
    rng = np.random.default_rng(seed)
    img = render(ctx.expected[idx], px, 4, ctx.palette)
    if torn_with is not None:
        other = render(ctx.expected[torn_with], px, 4, ctx.palette)
        cut = int(img.shape[0] * 0.55)
        img[cut:] = other[cut:]
    img = np.rot90(img, rot)
    if mirror:
        img = img[:, ::-1]
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    Wc, Hc = canvas
    side, cx, cy = fill * Hc, Wc / 2, Hc / 2
    dst = np.float32([[cx - side / 2, cy - side / 2], [cx + side / 2, cy - side / 2],
                      [cx + side / 2, cy + side / 2], [cx - side / 2, cy + side / 2]])
    dst += rng.uniform(-persp * side, persp * side, dst.shape).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    img = cv2.GaussianBlur(img, (0, 0), 0.4 * max(h / side, 1))  # 축소 전 앨리어싱 완화
    out = cv2.warpPerspective(img, Hm, (Wc, Hc), flags=cv2.INTER_LINEAR, borderValue=0)
    ys, xs = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    r2 = ((xs - cx) ** 2 + (ys - cy) ** 2) / (Wc / 2) ** 2
    out = cv2.remap(out, cx + (xs - cx) * (1 + barrel * r2), cy + (ys - cy) * (1 + barrel * r2),
                    cv2.INTER_LINEAR)
    out = cv2.GaussianBlur(out, (0, 0), blur)
    f = out.astype(np.float32) / 255 * np.array(gain, np.float32) + offset
    f += rng.normal(0, noise, f.shape).astype(np.float32)
    bgr = (np.clip(f, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def run_cases():
    ok_all = True
    print(f"{'케이스':<34}{'판정':>8}{'프레임':>7}{'셀px':>6}{'핀':>5}{'잔차':>7}"
          f"{'SER앵커':>9}{'SER국소':>9}{'SER정답':>9}{'테두리':>8}")
    ctxs = {pin: Context(1, 64, pin) for pin in ("checker", "cross")}
    ctxs["cmyg"] = Context(1, 64, "checker", palette="cmyg")
    cases = [
        ("기본 0°", "checker", dict(idx=5)),
        ("90° + 거울상", "checker", dict(idx=10, rot=1, mirror=True, seed=1)),
        ("180°, 셀 5px", "checker", dict(idx=20, rot=2, px=5, seed=2)),
        ("270°, 강한 배럴", "checker", dict(idx=33, rot=3, barrel=0.08, seed=3)),
        ("작게 (셀 약 3.4px)", "checker", dict(idx=40, fill=0.65, seed=4)),
        ("십자 핀", "cross", dict(idx=12, rot=1, seed=5)),
        ("찢김 (7|8)", "checker", dict(idx=7, torn_with=8, seed=6)),
        ("CMYG 팔레트 90°", "cmyg", dict(idx=21, rot=1, seed=7)),
    ]
    for name, pin, kw in cases:
        ctx = ctxs[pin]
        bgr = synth(ctx, **kw)
        r = analyze_image(bgr, ctx)
        if r["status"] == "nodetect":
            print(f"{name:<34}{'검출실패':>8}")
            ok_all = False
            continue
        print(f"{name:<34}{r['status']:>8}{r['best']:>7}{r['cell_px']:>6.2f}{r['pins_ok']:>5}"
              f"{r['pin_rms']:>7.3f}{r['ser_anchor']:>9.3%}{r['ser_dd']:>9.3%}{r['ser_oracle']:>9.3%}"
              f"{r['border_agree']:>8.1%}")
        if kw.get("torn_with") is not None:
            good = r["status"] == "torn"
        elif kw.get("fill", 0.8) < 0.7:
            good = r["best"] == kw["idx"]  # 저해상도는 식별만 요구
        else:
            good = r["status"] == "clean" and r["best"] == kw["idx"] and r["ser_dd"] < 0.005
        if not good:
            ok_all = False
            print(f"   ^ [실패] 기대: {'찢김' if kw.get('torn_with') is not None else '정상'}, 프레임 {kw['idx']}")
    print(f"[{'통과' if ok_all else '실패'}] 합성 이미지 검출·판정")
    return ok_all


if __name__ == "__main__":
    results = [check_js_parity(), run_cases()]
    sys.exit(0 if all(results) else 1)
