/* Event film — illustrations and icons (Notion-style line art).

   Used by scene.jsx. Everything here is a pure function of time: pass the
   film clock (or a local clock) as `t`, and draw-in progress as `p` (0..1).
   Nothing uses wall-clock or CSS animation, so ?t=SECONDS always lands on the
   same frame.

   Exports: palette, fonts, shadows, time helpers (shared with scene.jsx) and
   Engineer · Icon · CheckCircle · Checkbox · TeamVignette · RepeatVignette ·
   AskVignette · MiniTicket · DocGather */
import { createElement as h } from 'react'
import { Easing as E } from './engine'
import { clamp01, lerp } from '../shared'

// ---------- palette ----------
// Same roles as css/glance.css, frozen at the light theme (the film canvas stays light in
// dark mode): Tailwind neutral greys, the one light-theme blue #0070c9, Warning Pink for
// problems / privacy, and green / purple only for the kw / link feature kinds.
const INK = '#171717',
  MUT = '#737373',
  SUB = '#a3a3a3',
  FAINT = '#d4d4d4'
const HAIR = '#e5e5e5',
  WASH = '#f5f5f5',
  EDGE = '#dcdcdc',
  ACCENT = '#0070c9'
const PAPER = '#ededed'
const TONE = {
  gray: { fg: '#737373', bg: '#f0f0f0' },
  ink: { fg: INK, bg: '#f0f0f0' },
  blue: { fg: ACCENT, bg: '#e6f1fa' },
  pink: { fg: '#e83e8c', bg: '#fdecf4' },
  green: { fg: '#448361', bg: '#edf3ec' },
  purple: { fg: '#9065b0', bg: '#f4f0f7' },
}
const SANS =
  "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', Helvetica, Arial, sans-serif"
const SERIF = "'Songti SC', 'Noto Serif SC', 'Source Han Serif SC', STSong, Georgia, serif"
const MONO = "'SFMono-Regular', Menlo, Consolas, 'PingFang SC', monospace"
const SH_CARD =
  '0 0 0 1px rgba(15,15,15,.05), 0 3px 6px rgba(15,15,15,.05), 0 9px 24px rgba(15,15,15,.07)'
const SH_SOFT = '0 0 0 1px rgba(15,15,15,.07), 0 1px 3px rgba(15,15,15,.04)'

// ---------- time helpers ----------
const seg = (t, t0, t1, ease) => {
  const p = clamp01((t - t0) / (t1 - t0))
  return ease ? ease(p) : p
}
const frac = (x) => x - Math.floor(x)
// phase of a repeating cycle, 0..1
const cycle = (t, period, phase) => frac(t / period + (phase || 0))
// 0 → 1 → 0 over one cycle, smooth at both ends
const bell = (q) => Math.sin(Math.PI * clamp01(q))
const S = (v) => Math.max(0.001, v)

// ---------- drawing primitives ----------
const R = { strokeLinecap: 'round', strokeLinejoin: 'round' }
const ST = (color, w) => Object.assign({ fill: 'none', stroke: color, strokeWidth: w }, R)
// stroke draw-on; hides the round-cap dot at p = 0
const drawn = (p) => {
  const q = clamp01(p)
  return {
    pathLength: 1,
    strokeDasharray: 1,
    strokeDashoffset: 1 - q,
    strokeOpacity: q > 0.002 ? 1 : 0,
  }
}
const at = (x, y, s, r) =>
  `translate(${x} ${y})` +
  (r ? ` rotate(${r})` : '') +
  (s != null && s !== 1 ? ` scale(${S(s)})` : '')
const cubicAt = (a, b, c, d, s) => {
  const u = 1 - s
  return u * u * u * a + 3 * u * u * s * b + 3 * u * s * s * c + s * s * s * d
}
// a limb: ink outline with a fill core, joints rounded
const tube = (key, d, w, fill, sw) =>
  h(
    'g',
    { key: key },
    h(
      'path',
      Object.assign({ d: d, fill: 'none', stroke: INK, strokeWidth: w + 2 * (sw || 2.4) }, R)
    ),
    h('path', Object.assign({ d: d, fill: 'none', stroke: fill, strokeWidth: w }, R))
  )
const star4 = (r) =>
  `M0 ${-r}C${r * 0.12} ${-r * 0.12} ${r * 0.12} ${-r * 0.12} ${r} 0C${r * 0.12} ${r * 0.12} ${
    r * 0.12
  } ${r * 0.12} 0 ${r}C${-r * 0.12} ${r * 0.12} ${-r * 0.12} ${r * 0.12} ${-r} 0C${-r * 0.12} ${
    -r * 0.12
  } ${-r * 0.12} ${-r * 0.12} 0 ${-r}Z`

export {
  INK,
  MUT,
  SUB,
  FAINT,
  HAIR,
  WASH,
  EDGE,
  ACCENT,
  PAPER,
  TONE,
  SANS,
  SERIF,
  MONO,
  SH_CARD,
  SH_SOFT,
  seg,
  frac,
  cycle,
  bell,
  S,
  ST,
  drawn,
  at,
  cubicAt,
}

// =====================================================================
// THE ENGINEER — one character, two poses.
//   pose 'typing'  (film 0–10): headset on, typing; a help request lands
//                  on the screen at t = 1.5
//   pose 'relaxed' (film 19–24): leans back, lifts a coffee, the screen
//                  shows the answer
// `t` is local: seconds since the pose's scene began.
// =====================================================================
const SK = 2.4 // body stroke
const HIP = [150, 330]

function Clock({ t, speed }) {
  const m = t * 90 * speed,
    hr = 200 + t * 7.5 * speed
  return h(
    'g',
    { transform: 'translate(92 104)' },
    h('circle', { r: 25, fill: '#fff', stroke: INK, strokeWidth: 2.2 }),
    [0, 90, 180, 270].map((a) =>
      h(
        'line',
        Object.assign(
          { key: a, x1: 0, y1: -19, x2: 0, y2: -16, transform: `rotate(${a})` },
          ST(FAINT, 2)
        )
      )
    ),
    h(
      'line',
      Object.assign({ x1: 0, y1: 0, x2: 0, y2: -10, transform: `rotate(${hr})` }, ST(INK, 2.6))
    ),
    h(
      'line',
      Object.assign({ x1: 0, y1: 0, x2: 0, y2: -16, transform: `rotate(${m})` }, ST(INK, 1.8))
    ),
    h('circle', { r: 2.4, fill: INK })
  )
}

function Steam({ x, y, t, n, tone }) {
  const wisps = []
  for (let k = 0; k < (n || 2); k++) {
    const q = cycle(t, 2.6, k * 0.5)
    wisps.push(
      h(
        'path',
        Object.assign(
          {
            key: k,
            d: 'M0 0C-4 -6 4 -10 0 -16C-4 -22 3 -26 0 -31',
            transform: `translate(${x + k * 9} ${y - q * 12}) scale(${0.8 + 0.3 * q})`,
            opacity: 0.85 * bell(q),
          },
          ST(tone || FAINT, 1.9)
        )
      )
    )
  }
  return h('g', null, wisps)
}

function Plant({ t }) {
  const sway = 2.4 * Math.sin(t * 1.25)
  const leaf = (key, x, y, r, s, fill) =>
    h('path', {
      key: key,
      d: 'M0 0C5 -9 17 -13 28 -9C20 -1 9 3 0 0Z',
      transform: at(x, y, s, r),
      fill: fill,
      stroke: INK,
      strokeWidth: 2.2,
      strokeLinejoin: 'round',
    })
  return h(
    'g',
    null,
    h(
      'g',
      { transform: `rotate(${sway} 456 272)` },
      h(
        'path',
        Object.assign(
          { d: 'M456 272C455 250 450 236 441 222M456 272C458 252 466 240 476 232' },
          ST(INK, 2.2)
        )
      ),
      leaf('a', 448, 244, -128, 0.9, '#fff'),
      leaf('b', 441, 222, -100, 0.85, INK),
      leaf('c', 462, 250, -40, 0.9, '#fff'),
      leaf('d', 476, 232, -70, 0.75, '#fff')
    ),
    h('path', {
      d: 'M438 272H474L469 300H443Z',
      fill: '#fff',
      stroke: INK,
      strokeWidth: SK,
      strokeLinejoin: 'round',
    })
  )
}

function Mug({ x, y, tilt }) {
  return h(
    'g',
    { transform: at(x, y, 1, tilt || 0) },
    h('path', { d: 'M24 5C33 4 34 17 24 17', fill: 'none', stroke: INK, strokeWidth: SK }),
    h('path', {
      d: 'M0 0H25V18C25 24 21 27 16 27H9C4 27 0 24 0 18Z',
      fill: '#fff',
      stroke: INK,
      strokeWidth: SK,
      strokeLinejoin: 'round',
    }),
    h('path', Object.assign({ d: 'M5 9H13' }, ST(TONE.blue.fg, 2)))
  )
}

// the lid leans back: skew around its bottom edge (y = 296). Lines live in
// the skewed frame; round marks are placed with lid() so they stay round.
const LID_SKEW = 0.2
const lid = (x, y) => [x + LID_SKEW * (296 - y), y]

function TypingScreen({ t }) {
  const q = cycle(t, 5.2)
  const L = [
    [0, 30, ACCENT],
    [8, 48, FAINT],
    [8, 22, FAINT],
    [16, 40, FAINT],
    [0, 16, ACCENT],
  ]
  const fade = 1 - seg(q, 0.9, 1)
  let caret = null
  const rows = L.map(([ind, len, c], k) => {
    const p = seg(q, 0.04 + k * 0.16, 0.18 + k * 0.16)
    if (p > 0 && p < 1) caret = [274 + ind + len * p + 3, 236 + k * 10]
    return p > 0
      ? h(
          'line',
          Object.assign(
            {
              key: k,
              x1: 274 + ind,
              y1: 236 + k * 10,
              x2: 274 + ind + len * p,
              y2: 236 + k * 10,
              opacity: fade,
            },
            ST(c, 3.2)
          )
        )
      : null
  })
  if (!caret) caret = [274 + 19, 286]
  const blink = Math.floor(t * 2.4) % 2 === 0
  return h(
    'g',
    null,
    rows,
    h('rect', {
      x: caret[0],
      y: caret[1] - 4,
      width: 1.8,
      height: 8,
      fill: INK,
      opacity: blink ? 0.8 : 0,
    })
  )
}

function AnswerLines({ t }) {
  const ln = (k) => seg(t, 1.4 + k * 0.2, 1.9 + k * 0.2, E.easeOutCubic)
  return h(
    'g',
    null,
    h(
      'line',
      Object.assign(
        { x1: 290, y1: 274, x2: lerp(290, 334, ln(0)), y2: 274, strokeOpacity: ln(0) > 0 ? 1 : 0 },
        ST(FAINT, 3.2)
      )
    ),
    h(
      'line',
      Object.assign(
        { x1: 296, y1: 283, x2: lerp(296, 326, ln(1)), y2: 283, strokeOpacity: ln(1) > 0 ? 1 : 0 },
        ST(FAINT, 3.2)
      )
    )
  )
}

function AnswerMark({ t }) {
  const c = seg(t, 0.7, 1.4, E.easeOutBack)
  const tick = seg(t, 1.1, 1.6, E.easeOutCubic)
  const [x, y] = lid(312, 250)
  return h(
    'g',
    { transform: `translate(${x} ${y})` },
    h('circle', { r: 13 * S(c), fill: TONE.blue.bg, stroke: ACCENT, strokeWidth: 1.8 }),
    h('path', Object.assign({ d: 'M-6.5 0.5L-2 5 7 -5' }, ST(ACCENT, 2.6), drawn(tick)))
  )
}

function Laptop({ t, relaxed }) {
  const badge = relaxed ? 0 : seg(t, 1.5, 1.9, E.easeOutBack)
  const ring = relaxed || t < 1.5 ? 1 : cycle(t - 1.5, 1.5)
  const [bx, by] = lid(356, 222)
  return h(
    'g',
    null,
    h(
      'g',
      {
        transform: `translate(0 296) skewX(${
          (-Math.atan(LID_SKEW) * 180) / Math.PI
        }) translate(0 -296)`,
      },
      h('rect', {
        x: 262,
        y: 216,
        width: 100,
        height: 80,
        rx: 7,
        fill: '#fff',
        stroke: INK,
        strokeWidth: SK,
      }),
      h('rect', { x: 269, y: 223, width: 86, height: 64, rx: 3, fill: WASH }),
      relaxed ? h(AnswerLines, { t: t }) : h(TypingScreen, { t: t })
    ),
    relaxed ? h(AnswerMark, { t: t }) : null,
    badge > 0
      ? h(
          'g',
          { transform: `translate(${bx} ${by})` },
          h('circle', {
            r: 6 + 10 * ring,
            fill: 'none',
            stroke: TONE.pink.fg,
            strokeWidth: 1.5,
            opacity: 0.55 * (1 - ring),
          }),
          h('circle', { r: 7 * S(badge), fill: TONE.pink.fg, stroke: '#fff', strokeWidth: 2 })
        )
      : null,
    h('rect', {
      x: 236,
      y: 295,
      width: 142,
      height: 6,
      rx: 3,
      fill: PAPER,
      stroke: INK,
      strokeWidth: 2,
    })
  )
}

function Head({ t, relaxed }) {
  const blink = !relaxed && cycle(t, 3.7, 0.4) < 0.045
  const eye = relaxed
    ? h('path', Object.assign({ d: 'M202 184Q206.5 179.5 211 184' }, ST(INK, 2.3)))
    : blink
    ? h('path', Object.assign({ d: 'M203 183H210' }, ST(INK, 2.2)))
    : h('ellipse', { cx: 206.5, cy: 183, rx: 2.2, ry: 2.6, fill: INK })
  return h(
    'g',
    null,
    // face
    h('path', {
      d: 'M160 192C157 168 172 156 190 157C207 158 215 170 215 183L221.5 195.5Q222.5 200 216.5 200H215C215 208 210 215 200 215C188 216 174 212 166 204C162 200 160 196 160 192Z',
      fill: '#fff',
      stroke: INK,
      strokeWidth: SK,
      strokeLinejoin: 'round',
    }),
    // hair: curls on a solid cap
    h(
      'g',
      { fill: INK },
      h('path', {
        d: 'M157 198C148 178 152 154 175 149C196 144 214 154 216 170C209 171 201 167 195 171C191 179 183 181 177 181C173 188 169 196 163 203Z',
      }),
      [
        [163, 153, 9.5],
        [178, 146, 10],
        [195, 146, 9.5],
        [208, 154, 8],
        [153, 168, 9],
        [154, 185, 8],
        [186, 170, 6],
      ].map(([x, y, r], k) => h('circle', { key: k, cx: x, cy: y, r: r }))
    ),
    eye,
    h(
      'path',
      Object.assign(
        { d: relaxed ? 'M201 174Q206 170.5 211 173' : 'M201 175Q206 172 211 174.5' },
        ST(INK, 2)
      )
    ),
    h(
      'path',
      Object.assign(
        { d: relaxed ? 'M205.5 204.5Q211 209.5 216.5 203.5' : 'M208 205.5Q212 206.8 215.5 205' },
        ST(INK, 2)
      )
    ),
    // headset: band, ear cup, boom
    tube('band', 'M176 184C168 158 186 139 210 152', 3.2, '#fff', 1.9),
    h('rect', {
      x: 169,
      y: 181,
      width: 15,
      height: 23,
      rx: 7,
      fill: '#fff',
      stroke: INK,
      strokeWidth: SK,
    }),
    h('path', Object.assign({ d: 'M180 203Q185 218 202 214.5' }, ST(INK, 2.4))),
    h('rect', { x: 199, y: 211, width: 9, height: 6, rx: 3, fill: INK })
  )
}

function Engineer({ t, pose }) {
  const relaxed = pose === 'relaxed'
  const lean = relaxed ? -7 * seg(t, 0, 1.2, E.easeInOutCubic) : 0
  const nod = relaxed
    ? -5 * seg(t, 0.2, 1.4, E.easeInOutCubic) + 1.2 * Math.sin(t * 1.4)
    : 1.4 * Math.sin(t * 1.7)
  const breathe = 1 + 0.008 * Math.sin(t * 1.9)
  const tap = (ph) => -2.6 * Math.max(0, Math.sin(t * 15 + ph))

  // relaxed arm: from resting beside the laptop to holding the cup at the chin
  const lift = relaxed ? seg(t, 0.4, 1.5, E.easeInOutCubic) : 0
  const wrist = [lerp(238, 214, lift), lerp(290, 242, lift)]
  const elbow = [lerp(184, 196, lift), lerp(300, 298, lift)]
  const sip = relaxed ? 3 * Math.sin(Math.max(0, t - 1.5) * 1.6) * seg(t, 1.5, 2.1) : 0

  const body = [
    // chair back rides with the torso
    h('rect', {
      key: 'chair',
      x: 96,
      y: 222,
      width: 22,
      height: 108,
      rx: 11,
      fill: PAPER,
      stroke: INK,
      strokeWidth: SK,
      transform: 'rotate(-5 107 276)',
    }),
    tube('neck', 'M174 236L177 220', 12, '#fff'),
    h('path', {
      key: 'torso',
      d: 'M126 334C119 300 121 264 139 243C150 231 170 227 185 232C199 238 205 262 205 292L207 334Z',
      fill: '#fff',
      stroke: INK,
      strokeWidth: SK,
      strokeLinejoin: 'round',
      transform: `translate(0 334) scale(1 ${breathe}) translate(0 -334)`,
    }),
    h('path', Object.assign({ key: 'collar', d: 'M165 233L175 249 187 235' }, ST(INK, 2.2))),
    // head sits a touch low on the shoulders: short neck
    h(
      'g',
      { key: 'head', transform: `translate(1 8) rotate(${nod} 176 214)` },
      h(Head, { t: t, relaxed: relaxed })
    ),
  ]
  if (relaxed) {
    body.push(
      tube('arm', `M168 248L${elbow[0]} ${elbow[1]}L${wrist[0]} ${wrist[1]}`, 18, '#fff'),
      h(Mug, { key: 'mug', x: wrist[0] - 6, y: wrist[1] - 20, tilt: -sip }),
      h('ellipse', {
        key: 'hand',
        cx: wrist[0] - 2,
        cy: wrist[1] - 4,
        rx: 7,
        ry: 9,
        fill: '#fff',
        stroke: INK,
        strokeWidth: SK,
      }),
      lift > 0.95 ? h(Steam, { key: 'steam', x: wrist[0] + 2, y: wrist[1] - 24, t: t, n: 2 }) : null
    )
  } else {
    body.push(
      tube('arm', 'M168 248L180 298L250 292', 18, '#fff'),
      h('path', Object.assign({ key: 'cuff', d: 'M241 283.5L242.5 300.5' }, ST(INK, 2))),
      h('ellipse', {
        key: 'hand',
        cx: 257,
        cy: 291 + tap(0),
        rx: 9,
        ry: 6,
        fill: '#fff',
        stroke: INK,
        strokeWidth: SK,
      })
    )
  }

  const twinkle = relaxed ? seg(t, 1.6, 2.1, E.easeOutBack) * (0.8 + 0.2 * Math.sin(t * 5)) : 0
  return h(
    'svg',
    {
      width: 520,
      height: 430,
      viewBox: '0 0 520 430',
      style: { overflow: 'visible' },
      role: 'img',
      'aria-label': relaxed
        ? '售后工程师靠在椅背上端起咖啡，电脑上已经有了答案'
        : '戴着耳麦的售后工程师在电脑前处理支持工单',
    },
    h('circle', {
      cx: 268,
      cy: 232,
      r: 172 * lerp(0.94, 1, seg(t, 0, 1, E.easeOutCubic)),
      fill: WASH,
    }),
    h('path', Object.assign({ d: 'M40 404H488' }, ST(HAIR, 1.6))),
    h(Clock, { t: t, speed: relaxed ? 0.25 : 1 }),
    // chair base
    h(
      'g',
      null,
      h('path', Object.assign({ d: 'M152 342V380M118 382H188' }, ST(INK, SK))),
      [120, 152, 186].map((x) =>
        h('circle', { key: x, cx: x, cy: 391, r: 5.5, fill: '#fff', stroke: INK, strokeWidth: 2.2 })
      )
    ),
    h('rect', {
      x: 104,
      y: 331,
      width: 104,
      height: 11,
      rx: 5.5,
      fill: PAPER,
      stroke: INK,
      strokeWidth: SK,
    }),
    // legs: dark trousers, white shoes
    h(
      'path',
      Object.assign(
        {
          d: relaxed ? 'M146 324H236Q250 324 256 338L276 386' : 'M146 324H240Q252 324 252 336V386',
        },
        ST(INK, 25)
      )
    ),
    h('path', {
      d: relaxed
        ? 'M262 386H288Q301 387 302 398Q302 402 297 402H262Z'
        : 'M238 386H262Q275 387 276 398Q276 402 271 402H238Z',
      fill: '#fff',
      stroke: INK,
      strokeWidth: SK,
      strokeLinejoin: 'round',
    }),
    h('g', { transform: `rotate(${lean} ${HIP[0]} ${HIP[1]})` }, body),
    // desk
    h('path', {
      d: 'M44 300H480V309H44Z',
      fill: '#fff',
      stroke: INK,
      strokeWidth: SK,
      strokeLinejoin: 'round',
    }),
    h('path', Object.assign({ d: 'M70 309L64 400M454 309L460 400' }, ST(INK, SK))),
    h(Laptop, { t: t, relaxed: relaxed }),
    relaxed ? null : h(Mug, { x: 392, y: 273 }),
    relaxed ? null : h(Steam, { x: 399, y: 266, t: t, n: 2 }),
    h(Plant, { t: t }),
    twinkle > 0
      ? h('path', { d: star4(9), transform: at(244, 150, twinkle, t * 20), fill: TONE.blue.fg })
      : null,
    twinkle > 0
      ? h('path', {
          d: star4(5),
          transform: at(262, 172, twinkle * (0.7 + 0.3 * Math.sin(t * 6 + 1)), 0),
          fill: TONE.blue.fg,
        })
      : null
  )
}

// =====================================================================
// ICONS — one 24-unit grid, round joins, a single accent at most.
//   p   draw-in progress 0..1
//   t   seconds, for idle loops (0 freezes them)
//   on  0..1, how "active" the icon is (loops scale with it)
//   done 0..1, morph to a resolved state (question / shield)
// =====================================================================
const ICONS = {
  graph: ({ c, a, p, t, sw }) => {
    const N = [
      [5, 7],
      [16, 4.5],
      [8, 18],
      [19.5, 15.5],
    ]
    const EG = [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 3],
      [1, 2],
    ]
    const run = cycle(t, 1.8)
    const leg = run < 0.5 ? [N[0], N[1], run * 2] : [N[1], N[3], run * 2 - 1]
    return [
      EG.map(([i, j], k) =>
        h(
          'line',
          Object.assign(
            { key: 'e' + k, x1: N[i][0], y1: N[i][1], x2: N[j][0], y2: N[j][1] },
            ST(FAINT, sw),
            drawn(seg(p, k * 0.08, 0.4 + k * 0.08))
          )
        )
      ),
      p >= 1
        ? h('circle', {
            key: 'run',
            cx: lerp(leg[0][0], leg[1][0], leg[2]),
            cy: lerp(leg[0][1], leg[1][1], leg[2]),
            r: 1.1,
            fill: a,
          })
        : null,
      N.map(([x, y], i) =>
        h('circle', {
          key: 'n' + i,
          cx: x,
          cy: y,
          r: (i === 0 ? 2.8 : 2.3) * S(seg(p, 0.3 + i * 0.1, 0.6 + i * 0.1, E.easeOutBack)),
          fill: i === 0 ? a : c,
        })
      ),
    ]
  },
  search: ({ c, a, p, t, sw, on }) => {
    const dx = 0.7 * on * Math.cos(t * 3.2),
      dy = 0.7 * on * Math.sin(t * 3.2)
    const glint = cycle(t, 2.4)
    return h(
      'g',
      { transform: `translate(${dx} ${dy})` },
      h('circle', Object.assign({ cx: 10.5, cy: 10.5, r: 6.5 }, ST(c, sw), drawn(seg(p, 0, 0.6)))),
      h(
        'path',
        Object.assign({ d: 'M15.4 15.4L20.5 20.5' }, ST(c, sw * 1.15), drawn(seg(p, 0.5, 0.85)))
      ),
      h(
        'path',
        Object.assign(
          { d: 'M7.4 9.2A3.4 3.4 0 0 1 9.4 7.2', opacity: on * bell(glint * 1.6) },
          ST(a, sw * 0.8)
        )
      )
    )
  },
  // a peak with a flag planted on top — "where it gets hard"
  peak: ({ c, a, p, t, sw }) => {
    const wave = p >= 1 ? Math.sin(t * 5) : 0
    return [
      h(
        'path',
        Object.assign(
          { key: 'm', d: 'M2 20.5L9.5 8.5L13.5 14.5L16 11L22 20.5' },
          ST(c, sw),
          drawn(seg(p, 0, 0.6, E.easeInOutCubic))
        )
      ),
      h(
        'path',
        Object.assign(
          { key: 's', d: 'M7.4 11.8L9.5 13.2L11.4 11.4' },
          ST(c, sw * 0.8),
          drawn(seg(p, 0.45, 0.7))
        )
      ),
      h(
        'path',
        Object.assign({ key: 'pole', d: 'M9.5 8.5V2.5' }, ST(c, sw), drawn(seg(p, 0.55, 0.8)))
      ),
      h('path', {
        key: 'flag',
        d: `M9.5 2.5Q12 ${2.3 + wave * 0.6} 14.5 ${3.2 + wave * 0.3}L9.5 5.8Z`,
        fill: a,
        transform: `translate(9.5 4) scale(${S(
          seg(p, 0.75, 1, E.easeOutBack)
        )} 1) translate(-9.5 -4)`,
      }),
    ]
  },
  sparkle: ({ a, p, t, on }) => {
    const s1 = seg(p, 0, 0.7, E.easeOutBack) * (1 + 0.07 * on * Math.sin(t * 4))
    const s2 = seg(p, 0.35, 1, E.easeOutBack) * (0.75 + 0.25 * on * Math.sin(t * 5.5 + 1.2))
    return [
      h('path', {
        key: 'a',
        d: star4(8),
        transform: at(10.5, 13, s1, 8 * on * Math.sin(t * 1.3)),
        fill: a,
      }),
      h('path', { key: 'b', d: star4(3.8), transform: at(19.5, 4.8, s2), fill: a }),
    ]
  },
  ripple: ({ c, p, t, sw, on }) => {
    const rings = [0, 0.5].map((ph, k) => {
      const q = cycle(t, 1.6, ph)
      return h('circle', {
        key: k,
        cx: 12,
        cy: 12,
        r: 3 + 8 * q,
        fill: 'none',
        stroke: c,
        strokeWidth: sw * 0.8,
        opacity: on * (1 - q) * seg(p, 0.5, 1),
      })
    })
    return [
      h(
        'circle',
        Object.assign(
          { key: 's1', cx: 12, cy: 12, r: 6.5, opacity: 1 - on * 0.6 },
          ST(c, sw * 0.8),
          drawn(seg(p, 0.1, 0.7))
        )
      ),
      h(
        'circle',
        Object.assign(
          { key: 's2', cx: 12, cy: 12, r: 10, opacity: 0.45 * (1 - on) },
          ST(c, sw * 0.8),
          drawn(seg(p, 0.3, 0.9))
        )
      ),
      rings,
      h('circle', { key: 'c', cx: 12, cy: 12, r: 2.3 * S(seg(p, 0, 0.4, E.easeOutBack)), fill: c }),
    ]
  },
  text: ({ c, p, t, sw, on }) => {
    const hi =
      on * seg(cycle(t, 2.2), 0.1, 0.5, E.easeInOutCubic) * (1 - seg(cycle(t, 2.2), 0.85, 1))
    return [
      h('rect', {
        key: 'h',
        x: 3,
        y: 9.2,
        width: 16 * hi,
        height: 5.6,
        rx: 1.4,
        fill: c,
        opacity: 0.22,
      }),
      [
        [6, 17],
        [12, 19],
        [18, 12],
      ].map(([y, x2], k) =>
        h(
          'line',
          Object.assign(
            { key: k, x1: 4, y1: y, x2: x2, y2: y },
            ST(c, sw),
            drawn(seg(p, k * 0.15, 0.5 + k * 0.15))
          )
        )
      ),
    ]
  },
  tag: ({ c, p, t, sw, on }) =>
    h(
      'g',
      { transform: `rotate(${on * 7 * Math.sin(t * 3.4)} 8.5 8.5)` },
      h(
        'path',
        Object.assign(
          { d: 'M4 4H11.2L20.3 13.1Q21 13.8 20.3 14.5L14.5 20.3Q13.8 21 13.1 20.3L4 11.2Z' },
          ST(c, sw),
          drawn(seg(p, 0, 0.8))
        )
      ),
      h('circle', { cx: 8.5, cy: 8.5, r: 1.7 * S(seg(p, 0.5, 1, E.easeOutBack)), fill: c })
    ),
  nodes: ({ c, p, t, sw, on }) => {
    const N = [
      [5, 18],
      [12, 5.5],
      [19, 17],
    ]
    const q = cycle(t, 1.3)
    const run = q < 0.5 ? [N[0], N[1], q * 2] : [N[1], N[2], q * 2 - 1]
    return [
      [
        [0, 1],
        [1, 2],
        [0, 2],
      ].map(([i, j], k) =>
        h(
          'line',
          Object.assign(
            { key: 'e' + k, x1: N[i][0], y1: N[i][1], x2: N[j][0], y2: N[j][1], opacity: 0.55 },
            ST(c, sw * 0.85),
            drawn(seg(p, k * 0.12, 0.5 + k * 0.12))
          )
        )
      ),
      on > 0.05
        ? h('circle', {
            key: 'run',
            cx: lerp(run[0][0], run[1][0], run[2]),
            cy: lerp(run[0][1], run[1][1], run[2]),
            r: 1.4,
            fill: c,
            opacity: on,
          })
        : null,
      N.map(([x, y], i) =>
        h('circle', {
          key: 'n' + i,
          cx: x,
          cy: y,
          r: 2.6 * S(seg(p, 0.3 + i * 0.12, 0.65 + i * 0.12, E.easeOutBack)),
          fill: '#fff',
          stroke: c,
          strokeWidth: sw,
        })
      ),
    ]
  },
  bubble: ({ c, p, t, sw, on }) => [
    h(
      'path',
      Object.assign(
        {
          key: 'b',
          d: 'M6 4.5H18Q21 4.5 21 7.5V14Q21 17 18 17H11L6.5 20.5V17H6Q3 17 3 14V7.5Q3 4.5 6 4.5Z',
        },
        ST(c, sw),
        drawn(seg(p, 0, 0.8))
      )
    ),
    [8, 12, 16].map((x, k) =>
      h('circle', {
        key: k,
        cx: x,
        cy: 10.8 - on * 1.4 * Math.max(0, Math.sin(t * 7 - k * 0.9)),
        r: 1.25 * S(seg(p, 0.5 + k * 0.12, 0.8 + k * 0.12, E.easeOutBack)),
        fill: c,
      })
    ),
  ],
  stack: ({ c, p, t, sw, on }) => {
    const layer = (k, y) => {
      const q = seg(p, k * 0.2, 0.45 + k * 0.2, E.easeOutBack)
      const float = k === 0 ? -0.8 * on * (0.5 + 0.5 * Math.sin(t * 2.6)) : 0
      return h(
        'path',
        Object.assign(
          {
            key: k,
            d: `M12 ${y - 4.5}L20.5 ${y}L12 ${y + 4.5}L3.5 ${y}Z`,
            transform: `translate(0 ${(1 - q) * -5 + float})`,
            fill: k === 0 ? c : '#fff',
            fillOpacity: k === 0 ? 0.16 : 1,
            opacity: clamp01(q * 1.5),
          },
          { stroke: c, strokeWidth: sw, strokeLinejoin: 'round' }
        )
      )
    }
    return [layer(2, 17), layer(1, 12.5), layer(0, 8)]
  },
  question: ({ c, p, sw, done }) => [
    h(
      'path',
      Object.assign(
        {
          key: 'b',
          d: 'M12 3.5C17 3.5 20.5 6.8 20.5 11S17 18.5 12 18.5C11 18.5 10 18.4 9.1 18.1L4.5 20.5L5.5 16.3C4.3 14.9 3.5 13 3.5 11C3.5 6.8 7 3.5 12 3.5Z',
        },
        ST(c, sw),
        drawn(seg(p, 0, 0.8))
      )
    ),
    h(
      'g',
      { key: 'q', opacity: 1 - done },
      h(
        'path',
        Object.assign(
          {
            d: 'M9.6 9.1C9.8 7.8 10.8 7 12.1 7C13.5 7 14.5 7.9 14.5 9.1C14.5 10.8 12.2 10.9 12.2 12.6',
          },
          ST(c, sw),
          drawn(seg(p, 0.45, 0.9))
        )
      ),
      h('circle', { cx: 12.2, cy: 15.2, r: 1.05 * S(seg(p, 0.85, 1)), fill: c })
    ),
    h(
      'path',
      Object.assign({ key: 'ok', d: 'M8.5 11.2L11 13.6 15.8 8.6' }, ST(c, sw * 1.1), drawn(done))
    ),
  ],
  shield: ({ c, p, sw, done }) => [
    h(
      'path',
      Object.assign(
        {
          key: 's',
          d: 'M12 3L19.5 5.8V11.3C19.5 15.6 16.4 19.3 12 21C7.6 19.3 4.5 15.6 4.5 11.3V5.8Z',
        },
        ST(c, sw),
        drawn(seg(p, 0, 0.8))
      )
    ),
    h(
      'g',
      { key: 'x', opacity: 1 - done },
      h('path', Object.assign({ d: 'M12 8V12.6' }, ST(c, sw * 1.1), drawn(seg(p, 0.5, 0.85)))),
      h('circle', { cx: 12, cy: 15.6, r: 1.05 * S(seg(p, 0.85, 1)), fill: c })
    ),
    h(
      'path',
      Object.assign({ key: 'ok', d: 'M8.6 12.2L11 14.5 15.6 9.7' }, ST(c, sw * 1.1), drawn(done))
    ),
  ],
  bell: ({ c, a, p, t, sw, on }) => {
    const ring = on * 14 * Math.sin(t * 16) * Math.exp(-3 * (t % 1.8))
    return h(
      'g',
      { transform: `rotate(${ring} 12 4)` },
      h(
        'path',
        Object.assign(
          { d: 'M6.5 16.5V11C6.5 7.7 8.9 5.2 12 5.2S17.5 7.7 17.5 11V16.5L19 18H5Z' },
          ST(c, sw),
          drawn(p)
        )
      ),
      h(
        'path',
        Object.assign(
          { d: 'M10.2 20.2C10.6 21 11.2 21.4 12 21.4S13.4 21 13.8 20.2' },
          ST(c, sw),
          drawn(seg(p, 0.6, 1))
        )
      ),
      h('circle', {
        cx: 17.5,
        cy: 5.5,
        r: 2.6 * S(seg(p, 0.7, 1, E.easeOutBack)),
        fill: a,
        stroke: '#fff',
        strokeWidth: 1.2,
      })
    )
  },
  target: ({ c, a, p, t, sw }) => {
    const fly = seg(p, 0.45, 0.8, E.easeInCubic)
    const wob = p >= 0.8 ? 5 * Math.sin((p - 0.8) * 60) * (1 - seg(p, 0.8, 1)) : 0
    return [
      h(
        'circle',
        Object.assign({ key: 'r1', cx: 11, cy: 13, r: 8.5 }, ST(c, sw), drawn(seg(p, 0, 0.45)))
      ),
      h(
        'circle',
        Object.assign({ key: 'r2', cx: 11, cy: 13, r: 4.5 }, ST(c, sw), drawn(seg(p, 0.12, 0.5)))
      ),
      h('circle', { key: 'c', cx: 11, cy: 13, r: 1.4 * S(seg(p, 0.3, 0.55)), fill: c }),
      fly > 0
        ? h(
            'g',
            {
              key: 'arr',
              transform: `translate(${11 + (1 - fly) * 9} ${13 - (1 - fly) * 9}) rotate(${wob})`,
            },
            h(
              'path',
              Object.assign(
                { d: 'M0 0L9 -9M0 0L3.6 -0.3M0 0L0.3 -3.6M7 -9.6L9.6 -9.6 9.6 -7' },
                ST(a, sw)
              )
            )
          )
        : null,
    ]
  },
  stopwatch: ({ c, a, p, t, sw }) => {
    const spin = 360 * 2.2 * E.easeOutCubic(seg(p, 0.3, 1)) + 40
    return [
      h(
        'circle',
        Object.assign({ key: 'o', cx: 12, cy: 13.5, r: 8 }, ST(c, sw), drawn(seg(p, 0, 0.5)))
      ),
      h(
        'path',
        Object.assign(
          { key: 'k', d: 'M10 3H14M12 3V5.5M18.2 6.6L19.6 5.2' },
          ST(c, sw),
          drawn(seg(p, 0.3, 0.6))
        )
      ),
      h(
        'line',
        Object.assign(
          { key: 'h', x1: 12, y1: 13.5, x2: 12, y2: 8.5, transform: `rotate(${spin} 12 13.5)` },
          ST(a, sw)
        )
      ),
      h('circle', { key: 'c', cx: 12, cy: 13.5, r: 1.3, fill: a }),
    ]
  },
  // a Notion quote block: accent bar, lines of text, one line highlighted
  quote: ({ c, a, p, t, sw }) => [
    h('rect', {
      key: 'hl',
      x: 7.5,
      y: 9.6,
      width:
        12 *
        seg(cycle(t, 3), 0.1, 0.5, E.easeInOutCubic) *
        (1 - seg(cycle(t, 3), 0.85, 1)) *
        seg(p, 0.8, 1),
      height: 4.8,
      rx: 1.2,
      fill: a,
      opacity: 0.22,
    }),
    h(
      'path',
      Object.assign({ key: 'bar', d: 'M4 4.5V19.5' }, ST(a, sw * 1.5), drawn(seg(p, 0, 0.4)))
    ),
    [
      [7, 18],
      [12, 19.5],
      [17, 14],
    ].map(([y, x2], k) =>
      h(
        'path',
        Object.assign(
          { key: k, d: `M8.5 ${y}H${x2}` },
          ST(c, sw),
          drawn(seg(p, 0.25 + k * 0.15, 0.6 + k * 0.15))
        )
      )
    ),
  ],
  moon: ({ c, a, p, t, sw }) => {
    const tw = (ph) => 0.6 + 0.4 * Math.sin(t * 3 + ph)
    return [
      h(
        'path',
        Object.assign(
          {
            key: 'm',
            d: 'M16.5 16.2A7.8 7.8 0 0 1 8.3 4.6A8 8 0 1 0 19.4 13.2A7.6 7.6 0 0 1 16.5 16.2Z',
          },
          ST(c, sw),
          drawn(seg(p, 0, 0.8))
        )
      ),
      h('path', {
        key: 's1',
        d: star4(2.6),
        transform: at(18.5, 5.5, seg(p, 0.6, 1, E.easeOutBack) * tw(0)),
        fill: a,
      }),
      h('path', {
        key: 's2',
        d: star4(1.6),
        transform: at(21.5, 9.5, seg(p, 0.7, 1, E.easeOutBack) * tw(2)),
        fill: a,
      }),
    ]
  },
}

function Icon(props) {
  const {
    name,
    size = 24,
    color = INK,
    accent = ACCENT,
    p = 1,
    t = 0,
    sw = 1.6,
    on = 1,
    done = 0,
    style,
  } = props
  const fn = ICONS[name]
  return h(
    'svg',
    {
      width: size,
      height: size,
      viewBox: '0 0 24 24',
      style: Object.assign({ overflow: 'visible', display: 'block', flexShrink: 0 }, style),
    },
    fn({ c: color, a: accent, p: clamp01(p), t: t, sw: sw, on: clamp01(on), done: clamp01(done) })
  )
}

// filled circle with a drawn tick and a one-shot ring burst
function CheckCircle({ size = 20, color = TONE.blue.fg, p = 1 }) {
  const burst = seg(p, 0.45, 1)
  return h(
    'svg',
    {
      width: size,
      height: size,
      viewBox: '0 0 20 20',
      style: { flexShrink: 0, overflow: 'visible' },
    },
    burst > 0 && burst < 1
      ? h('circle', {
          cx: 10,
          cy: 10,
          r: 9 + 5 * burst,
          fill: 'none',
          stroke: color,
          strokeWidth: 1.2,
          opacity: 0.6 * (1 - burst),
        })
      : null,
    h('circle', { cx: 10, cy: 10, r: 9 * S(E.easeOutBack(seg(p, 0, 0.55))), fill: color }),
    h(
      'path',
      Object.assign({ d: 'M6 10.4L8.8 13 14 7.4' }, ST('#fff', 1.9), drawn(seg(p, 0.35, 0.85)))
    )
  )
}

// the Notion to-do checkbox
function Checkbox({ size = 22, p = 0 }) {
  const pop = p > 0 && p < 1 ? 1 + 0.18 * Math.sin(Math.PI * p) : 1
  return h(
    'svg',
    {
      width: size,
      height: size,
      viewBox: '0 0 22 22',
      style: { overflow: 'visible', display: 'block' },
    },
    h(
      'g',
      { transform: `translate(11 11) scale(${pop}) translate(-11 -11)` },
      h('rect', {
        x: 1,
        y: 1,
        width: 20,
        height: 20,
        rx: 4,
        fill: '#fff',
        stroke: p > 0.3 ? ACCENT : FAINT,
        strokeWidth: 1.5,
      }),
      h('rect', {
        x: 1,
        y: 1,
        width: 20,
        height: 20,
        rx: 4,
        fill: ACCENT,
        transform: `translate(11 11) scale(${S(E.easeOutBack(seg(p, 0, 0.6)))}) translate(-11 -11)`,
      }),
      h(
        'path',
        Object.assign({ d: 'M6 11.3L9.4 14.5 16 7.8' }, ST('#fff', 2), drawn(seg(p, 0.3, 0.9)))
      )
    )
  )
}

// =====================================================================
// VIGNETTES for the three closing lines (each ~320 × 130; p runs 0..1
// across the first ~2.2 s, t keeps idle loops alive while the line holds)
// =====================================================================
function Person({ x, y, s }) {
  return h(
    'g',
    { transform: at(x, y, s) },
    h('path', {
      d: 'M-17 30C-17 18 -9 12 0 12S17 18 17 30',
      fill: '#fff',
      stroke: INK,
      strokeWidth: 2.2,
      strokeLinejoin: 'round',
    }),
    h('circle', { cx: 0, cy: 0, r: 9, fill: '#fff', stroke: INK, strokeWidth: 2.2 })
  )
}

function TeamVignette({ p, t }) {
  const K = [
    [128, 30],
    [160, 14],
    [192, 30],
    [160, 48],
  ]
  const KE = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [1, 3],
  ]
  const PX = [96, 160, 224]
  const leave = seg(p, 0.62, 0.92, E.easeInOutCubic)
  const kept = seg(p, 0.78, 0.95, E.easeOutBack)
  const ring = cycle(t, 1.6)
  return h(
    'svg',
    { width: 320, height: 132, viewBox: '0 0 320 132', style: { overflow: 'visible' } },
    KE.map(([i, j], k) =>
      h(
        'line',
        Object.assign(
          { key: 'k' + k, x1: K[i][0], y1: K[i][1], x2: K[j][0], y2: K[j][1] },
          ST(EDGE, 1.6),
          drawn(seg(p, 0.35 + k * 0.03, 0.55 + k * 0.03))
        )
      )
    ),
    PX.map((x, i) => {
      const fade = i === 2 ? 1 - leave : 1
      return h(
        'line',
        Object.assign(
          {
            key: 'l' + i,
            x1: x,
            y1: 84,
            x2: lerp(x, K[i][0], 0.82),
            y2: lerp(84, K[i][1], 0.82),
            opacity: fade,
            strokeDasharray: '3 4',
          },
          ST(FAINT, 1.5),
          { strokeDashoffset: -t * 8, strokeOpacity: seg(p, 0.28, 0.4) }
        )
      )
    }),
    K.map(([x, y], i) => {
      const lit = i === 2 ? kept : 0
      return h(
        'g',
        { key: 'n' + i },
        lit > 0
          ? h('circle', {
              cx: x,
              cy: y,
              r: 6 + 10 * ring,
              fill: 'none',
              stroke: ACCENT,
              strokeWidth: 1.2,
              opacity: lit * (1 - ring),
            })
          : null,
        h('circle', {
          cx: x,
          cy: y,
          r: 6 * S(seg(p, 0.3 + i * 0.06, 0.52 + i * 0.06, E.easeOutBack)),
          fill: lit > 0.5 ? ACCENT : INK,
        })
      )
    }),
    PX.map((x, i) => {
      const q = seg(p, i * 0.08, 0.26 + i * 0.08, E.easeOutBack)
      const dx = i === 2 ? 46 * leave : 0
      return h(
        'g',
        { key: 'p' + i, opacity: i === 2 ? 1 - leave : 1 },
        Person({ x: x + dx, y: 96, s: q })
      )
    })
  )
}

function Bug({ x, y, s, color, fill }) {
  const leg = (k) =>
    h(
      'path',
      Object.assign(
        {
          key: k,
          d:
            k < 3
              ? `M-7 ${-4 + k * 5}L-13 ${-7 + k * 6}`
              : `M7 ${-4 + (k - 3) * 5}L13 ${-7 + (k - 3) * 6}`,
        },
        ST(color, 2)
      )
    )
  return h(
    'g',
    { transform: at(x, y, s) },
    [0, 1, 2, 3, 4, 5].map(leg),
    h('path', Object.assign({ d: 'M-3 -13L-6 -18M3 -13L6 -18' }, ST(color, 2))),
    h('ellipse', { cx: 0, cy: 1, rx: 8, ry: 10, fill: fill, stroke: color, strokeWidth: 2.2 }),
    h('circle', { cx: 0, cy: -11, r: 4.5, fill: color }),
    h('line', Object.assign({ x1: 0, y1: -8, x2: 0, y2: 11 }, ST(color, 1.6)))
  )
}

function RepeatVignette({ p, t }) {
  const A = [84, 70],
    B = [236, 70]
  const bIn = seg(p, 0.3, 0.46, E.easeOutBack)
  const shake = p > 0.3 && p < 0.55 ? 3 * Math.sin(p * 90) * (1 - seg(p, 0.4, 0.55)) : 0
  const arc = seg(p, 0.5, 0.76, E.easeInOutCubic)
  const seen = seg(p, 0.74, 0.9, E.easeOutCubic)
  const ring = cycle(t, 1.5)
  const pink = TONE.pink.fg
  // the older bug warms from grey to pink once it is linked
  const aCol = seen > 0.5 ? pink : SUB
  return h(
    'svg',
    { width: 320, height: 132, viewBox: '0 0 320 132', style: { overflow: 'visible' } },
    h('path', Object.assign({ d: 'M34 108H286' }, ST(HAIR, 1.6), drawn(seg(p, 0, 0.3)))),
    [84, 160, 236].map((x, k) =>
      h(
        'line',
        Object.assign(
          {
            key: k,
            x1: x,
            y1: 104,
            x2: x,
            y2: 112,
            opacity: seg(p, 0.1 + k * 0.08, 0.25 + k * 0.08),
          },
          ST(FAINT, 1.6)
        )
      )
    ),
    h(
      'path',
      Object.assign(
        { d: `M${B[0]} ${B[1] - 26}Q160 -8 ${A[0]} ${A[1] - 26}`, strokeDasharray: '4 5' },
        ST(pink, 1.6),
        { strokeOpacity: arc > 0 ? 0.9 : 0, clipPath: 'url(#repClip)' }
      )
    ),
    h(
      'defs',
      null,
      h(
        'clipPath',
        { id: 'repClip' },
        h('rect', { x: lerp(B[0] + 10, A[0] - 10, arc), y: -30, width: 320, height: 120 })
      )
    ),
    [A, B].map(([x, y], k) =>
      seen > 0
        ? h('circle', {
            key: 'r' + k,
            cx: x,
            cy: y,
            r: 20 + 10 * ring,
            fill: 'none',
            stroke: pink,
            strokeWidth: 1.2,
            opacity: seen * (1 - ring) * 0.8,
          })
        : null
    ),
    [A, B].map(([x, y], k) =>
      seen > 0
        ? h('circle', { key: 'h' + k, cx: x, cy: y, r: 20, fill: TONE.pink.bg, opacity: seen })
        : null
    ),
    h(Bug, { x: A[0], y: A[1], s: seg(p, 0.02, 0.2, E.easeOutBack), color: aCol, fill: '#fff' }),
    h(Bug, { x: B[0] + shake, y: B[1], s: bIn, color: pink, fill: '#fff' })
  )
}

function AskVignette({ p, t }) {
  const card = seg(p, 0, 0.22, E.easeOutBack)
  const bub = seg(p, 0.18, 0.34, E.easeOutBack)
  const line = seg(p, 0.34, 0.56, E.easeInOutCubic)
  const G = [
    [214, 52],
    [244, 32],
    [274, 56],
    [248, 86],
    [214, 90],
  ]
  const GE = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 0],
    [0, 3],
  ]
  const lit = seg(p, 0.6, 0.76, E.easeOutBack)
  const back = seg(p, 0.78, 0.95, E.easeOutBack)
  const run = cycle(t, 1.4)
  return h(
    'svg',
    { width: 320, height: 132, viewBox: '0 0 320 132', style: { overflow: 'visible' } },
    h(
      'g',
      { transform: `translate(72 70) scale(${S(card)}) translate(-72 -70)` },
      h('rect', {
        x: 32,
        y: 42,
        width: 80,
        height: 58,
        rx: 7,
        fill: '#fff',
        stroke: INK,
        strokeWidth: 2.2,
      }),
      h('circle', { cx: 44, cy: 55, r: 3.4, fill: TONE.pink.fg }),
      h('line', Object.assign({ x1: 54, y1: 55, x2: 96, y2: 55 }, ST(INK, 2.4))),
      h('line', Object.assign({ x1: 44, y1: 71, x2: 100, y2: 71 }, ST(FAINT, 2.4))),
      h('line', Object.assign({ x1: 44, y1: 84, x2: 82, y2: 84 }, ST(FAINT, 2.4)))
    ),
    h(
      'g',
      { transform: `translate(108 30) scale(${S(bub)})` },
      h('path', {
        d: 'M-4 -18H30Q36 -18 36 -12V2Q36 8 30 8H6L-2 14V8H-4Q-10 8 -10 2V-12Q-10 -18 -4 -18Z',
        fill: '#fff',
        stroke: ACCENT,
        strokeWidth: 2,
      }),
      [4, 13, 22].map((x, k) =>
        h('circle', {
          key: k,
          cx: x,
          cy: -5 - 1.8 * Math.max(0, Math.sin(t * 7 - k * 0.9)),
          r: 2.2,
          fill: ACCENT,
        })
      )
    ),
    h(
      'line',
      Object.assign(
        {
          x1: 120,
          y1: 71,
          x2: lerp(120, 202, line),
          y2: 71,
          strokeDasharray: '4 5',
          strokeOpacity: line > 0 ? 1 : 0,
        },
        ST(ACCENT, 1.8)
      )
    ),
    line >= 1
      ? h('circle', { cx: lerp(120, 202, run), cy: 71, r: 2.6, fill: ACCENT, opacity: bell(run) })
      : null,
    GE.map(([i, j], k) =>
      h(
        'line',
        Object.assign(
          { key: 'e' + k, x1: G[i][0], y1: G[i][1], x2: G[j][0], y2: G[j][1] },
          ST(lit > 0.3 && (i === 0 || j === 0) ? TONE.blue.fg : EDGE, 1.6),
          drawn(seg(p, 0.42 + k * 0.03, 0.6 + k * 0.03))
        )
      )
    ),
    G.map(([x, y], i) =>
      h('circle', {
        key: 'n' + i,
        cx: x,
        cy: y,
        r: (i === 0 ? 7 : 5.5) * S(seg(p, 0.44 + i * 0.04, 0.6 + i * 0.04, E.easeOutBack)),
        fill: i === 0 && lit > 0.5 ? ACCENT : INK,
      })
    ),
    lit > 0
      ? h('circle', {
          cx: G[0][0],
          cy: G[0][1],
          r: 7 + 14 * cycle(t, 1.6),
          fill: 'none',
          stroke: ACCENT,
          strokeWidth: 1.3,
          opacity: lit * (1 - cycle(t, 1.6)),
        })
      : null,
    back > 0
      ? h(
          'g',
          { transform: at(290, 20, back) },
          h('circle', { r: 13, fill: TONE.blue.fg }),
          h(
            'path',
            Object.assign({ d: 'M-5.5 0.5L-1.5 4.5 6 -4' }, ST('#fff', 2.6), drawn(seg(p, 0.86, 1)))
          )
        )
      : null
  )
}

// a tiny ticket card, centred on (x, y)
function MiniTicket({ x, y, s, o, key }) {
  return h(
    'g',
    { key: key, transform: at(x, y, s), opacity: o },
    h('rect', {
      x: -14,
      y: -10,
      width: 28,
      height: 20,
      rx: 4,
      fill: '#fff',
      stroke: INK,
      strokeWidth: 1.6,
    }),
    h('circle', { cx: -8, cy: -4, r: 1.8, fill: FAINT }),
    h('line', Object.assign({ x1: -3.5, y1: -4, x2: 8, y2: -4 }, ST(INK, 1.6))),
    h('line', Object.assign({ x1: -8, y1: 3.5, x2: 5, y2: 3.5 }, ST(FAINT, 1.6)))
  )
}

// bento loop: seven tickets link up, fold into one page, the page writes itself
function DocGather({ t }) {
  const P = [
    [20, 20],
    [58, 8],
    [44, 42],
    [86, 30],
    [26, 64],
    [70, 62],
    [104, 54],
  ]
  const EG = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [2, 4],
    [3, 5],
    [4, 5],
    [5, 6],
    [3, 6],
  ]
  const q = cycle(t, 6)
  const link = seg(q, 0.05, 0.25)
  const fold = seg(q, 0.32, 0.55, E.easeInOutCubic)
  const write = (k) => seg(q, 0.55 + k * 0.08, 0.68 + k * 0.08, E.easeOutCubic)
  const reset = seg(q, 0.9, 1)
  const dst = [62, 126]
  const pt = (i) => [lerp(P[i][0], dst[0], fold), lerp(P[i][1], dst[1] - 18, fold)]
  return h(
    'svg',
    { width: 130, height: 156, style: { display: 'block', overflow: 'visible' } },
    EG.map(([i, j], k) => {
      const a = pt(i),
        b = pt(j)
      return h(
        'line',
        Object.assign(
          { key: 'e' + k, x1: a[0], y1: a[1], x2: b[0], y2: b[1], opacity: 1 - fold },
          ST(EDGE, 1.3),
          drawn(link)
        )
      )
    }),
    P.map((_, i) => {
      const a = pt(i)
      return h('circle', {
        key: 'p' + i,
        cx: a[0],
        cy: a[1],
        r: 4.5 * (1 - 0.6 * fold),
        fill: INK,
        opacity: seg(q, 0, 0.06) * (1 - seg(fold, 0.8, 1)),
      })
    }),
    h(
      'g',
      { transform: at(dst[0], dst[1], 1) },
      h('path', {
        d: 'M-22 -24H12L22 -14V24H-22Z',
        fill: '#fff',
        stroke: INK,
        strokeWidth: 1.6,
        strokeLinejoin: 'round',
      }),
      h('path', Object.assign({ d: 'M12 -24V-14H22' }, ST(INK, 1.6))),
      [0, 1, 2, 3].map((k) =>
        h(
          'line',
          Object.assign(
            {
              key: k,
              x1: -14,
              y1: -10 + k * 9,
              x2: -14 + [26, 20, 24, 14][k] * write(k) * (1 - reset),
              y2: -10 + k * 9,
              strokeOpacity: write(k) * (1 - reset) > 0 ? 1 : 0,
            },
            ST(k === 0 ? ACCENT : FAINT, 2.2)
          )
        )
      )
    )
  )
}

export {
  Engineer,
  Icon,
  CheckCircle,
  Checkbox,
  TeamVignette,
  RepeatVignette,
  AskVignette,
  MiniTicket,
  DocGather,
}
