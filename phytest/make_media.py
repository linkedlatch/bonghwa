#!/usr/bin/env python3
"""송신 패턴을 PNG(정지) / GIF(재생)로 만든다. 브라우저 대신 이미지 뷰어로 띄울 때 사용.

    python3 make_media.py --px 3 4 5 --out media/

GIF는 무손실이다 (색 5개: 4색 + 흑색 여백). 프레임 지연은 1/100초 단위라
15 fps(66.7 ms)는 70/70/60 ms를 번갈아 써서 평균을 맞춘다. 10 fps(100 ms)는 정확하다.
"""
import argparse
import os

import numpy as np
from PIL import Image

from vdsp_layout import Layout, gen_frame, palette_rgb

BLACK = 4


def frame_indices(cells, px, quiet):
    """셀 값(0~3) → 팔레트 인덱스 이미지. 여백은 BLACK."""
    img = np.pad(cells, quiet, constant_values=BLACK).astype(np.uint8)
    return np.repeat(np.repeat(img, px, 0), px, 1)


def to_image(idx, palette="rgbw"):
    im = Image.fromarray(idx, "P")
    pal = list(palette_rgb(palette).ravel()) + [0, 0, 0]
    im.putpalette(pal + [0] * (768 - len(pal)))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--px", type=int, nargs="+", default=[3, 4, 5], help="셀당 화면 픽셀 (여러 개 가능)")
    ap.add_argument("--quiet", type=int, default=6, help="흑색 여백 셀 수")
    ap.add_argument("--n", type=int, default=64, help="재생 시퀀스 길이 (분석기 --n과 같아야 함)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pin", default="checker", choices=["checker", "cross"])
    ap.add_argument("--counter", default="bin", choices=["bin", "gray"])
    ap.add_argument("--palette", default="rgbw", choices=["rgbw", "cmyg"])
    ap.add_argument("--fps", type=int, nargs="+", default=[10, 15], choices=[10, 15])
    ap.add_argument("--out", default="media")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    L = Layout(args.pin)
    frames = [gen_frame(L, i, args.seed, args.counter) for i in range(args.n)]
    tag = ("" if args.pin == "checker" else f"_{args.pin}") + ("" if args.palette == "rgbw" else f"_{args.palette}")

    for px in args.px:
        side = (192 + 2 * args.quiet) * px
        png = os.path.join(args.out, f"static_{px}px{tag}.png")
        to_image(frame_indices(frames[0], px, args.quiet), args.palette).save(png, optimize=True)
        print(f"{png}  ({side}x{side})")
        imgs = [to_image(frame_indices(c, px, args.quiet), args.palette) for c in frames]
        for fps in args.fps:
            dur = [100] * args.n if fps == 10 else [(70, 70, 60)[i % 3] for i in range(args.n)]
            gif = os.path.join(args.out, f"anim_{fps}fps_{px}px{tag}.gif")
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=dur, loop=0,
                         optimize=False, disposal=1)
            print(f"{gif}  ({side}x{side}, {os.path.getsize(gif) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
