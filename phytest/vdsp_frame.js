/*
 * VDSP v1.9 물리 계층 테스트 패턴 (레이아웃 + 프레임 생성).
 * vdsp_layout.py와 비트 단위로 같은 결과를 내야 한다 (selftest.py가 비교).
 *
 * 셀 값: 0=청(B) 1=적(R) 2=녹(G) 3=백(W)  (spec 2.1)
 * 데이터 셀은 PRBS/RS 없이 xorshift32 난수로 채운다 (물리 계층 오인식률 측정용).
 */
(function (root) {
  'use strict';

  const S = 192;
  const BLUE = 0, RED = 1, GREEN = 2, WHITE = 3;
  const REGION = { DATA: 0, BORDER: 1, ANCHOR: 2, SEPARATOR: 3, PIN: 4 };
  const PALETTE = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 255]];
  // 셀 값(역할, 휘도 순) → 표시 색. vdsp_layout.py의 PALETTES와 같아야 한다
  const PALETTES = {
    rgbw: PALETTE,
    cmyg: [[255, 0, 255], [0, 255, 0], [0, 255, 255], [255, 255, 0]],
  };
  const ANCHOR_PATTERN = [[BLUE, GREEN], [WHITE, RED]]; // [qy][qx]
  const GRAY2 = [0, 1, 3, 2];

  const inAnchor = (c) => (c >= 3 && c <= 10) || (c >= 181 && c <= 188);
  const inSep = (c) => (c >= 2 && c <= 11) || (c >= 180 && c <= 189);
  const anchorOrigin = (c) => (c <= 11 ? 3 : 181);
  const anchorColor = (ax, ay) => ANCHOR_PATTERN[ay >= 4 ? 1 : 0][ax >= 4 ? 1 : 0];
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

  function pinColor(dx, dy, style) {
    if (style === 'cross') return dx === 0 || dy === 0 ? GREEN : BLUE;
    return ((dx + dy) & 1) === 0 ? GREEN : BLUE; // spec 2.4 체커보드
  }

  function buildLayout(pinStyle) {
    pinStyle = pinStyle || 'checker';
    const region = new Uint8Array(S * S);
    const fixed = new Int8Array(S * S).fill(-1);
    const ring = new Int8Array(S * S).fill(-1);
    const ringIdx = new Int32Array(S * S).fill(-1);
    const dataCells = [];

    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const p = y * S + x;
        if (x < 2 || y < 2 || x >= S - 2 || y >= S - 2) {
          region[p] = REGION.BORDER;
          const k = Math.min(x, y, S - 1 - x, S - 1 - y);
          const last = S - 1 - k, L = S - 2 * k;
          let i;
          if (y === k) i = x - k;
          else if (x === last) i = (L - 1) + (y - k);
          else if (y === last) i = 2 * (L - 1) + (last - x);
          else i = 3 * (L - 1) + (last - y);
          ring[p] = k;
          ringIdx[p] = i;
        } else if (inAnchor(x) && inAnchor(y)) {
          region[p] = REGION.ANCHOR;
          fixed[p] = anchorColor(x - anchorOrigin(x), y - anchorOrigin(y));
        } else if (inSep(x) && inSep(y)) {
          region[p] = REGION.SEPARATOR;
          const ox = anchorOrigin(x), oy = anchorOrigin(y);
          const c = anchorColor(clamp(x, ox, ox + 7) - ox, clamp(y, oy, oy + 7) - oy);
          fixed[p] = (c === BLUE || c === RED) ? WHITE : BLUE;
        } else {
          const i = Math.floor(x / 16 + 0.5), j = Math.floor(y / 16 + 0.5);
          if (i >= 1 && i <= 11 && j >= 1 && j <= 11 &&
              Math.abs(x - 16 * i) <= 1 && Math.abs(y - 16 * j) <= 1) {
            region[p] = REGION.PIN;
            fixed[p] = pinColor(x - 16 * i, y - 16 * j, pinStyle);
          } else {
            region[p] = REGION.DATA;
            dataCells.push(p);
          }
        }
      }
    }
    return { pinStyle, region, fixed, ring, ringIdx, dataCells: Int32Array.from(dataCells) };
  }

  function frameSeed(idx, seed) {
    const s = (Math.imul(0x9E3779B9 | 0, idx + 1) ^ seed) >>> 0;
    return s === 0 ? 1 : s;
  }

  function xorshift(x) {
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    return x >>> 0;
  }

  function counterValue(idx, mode) {
    return mode === 'gray' ? GRAY2[idx & 3] : idx & 3;
  }

  /** 프레임 idx의 셀 값 배열 (길이 S*S, 래스터 순서). */
  function genFrame(layout, idx, opts) {
    opts = opts || {};
    const seed = (opts.seed === undefined ? 1 : opts.seed) >>> 0;
    const ctr = counterValue(idx, opts.counter || 'bin');
    const cells = new Uint8Array(S * S);

    for (let p = 0; p < S * S; p++) {
      const r = layout.region[p];
      if (r === REGION.BORDER) {
        const k = layout.ring[p];
        const bit = k === 1 ? (ctr & 1) : ((ctr >> 1) & 1); // spec 2.5: 안쪽 링 = bit 0
        const first = (layout.ringIdx[p] & 1) === 0;
        cells[p] = (bit === 1) === first ? WHITE : BLUE; // 1 = [백, 청], 0 = [청, 백]
      } else if (r !== REGION.DATA) {
        cells[p] = layout.fixed[p];
      }
    }

    let x = frameSeed(idx, seed);
    for (let i = 0; i < 8; i++) x = xorshift(x);
    const d = layout.dataCells;
    for (let n = 0; n < d.length; n += 16) {
      x = xorshift(x);
      for (let k = 0; k < 16 && n + k < d.length; k++) {
        cells[d[n + k]] = (x >>> (30 - 2 * k)) & 3;
      }
    }
    return cells;
  }

  const api = { S, BLUE, RED, GREEN, WHITE, REGION, PALETTE, PALETTES, buildLayout, genFrame, counterValue };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.VDSP = api;
})(this);
