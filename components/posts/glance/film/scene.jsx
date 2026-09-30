/* Event film — 120 s, 1280×720, Notion-style (warm ink on white, soft tags,
   quiet shadows). Pure captions, no voice-over, no headers or numbering.

   One persistent knowledge graph carries the film: it forms out of a sea of
   closed tickets (10–19), is rebuilt node by node from one ticket (33–40),
   and is searched by four channels (41.5–60).

     0–10    a support engineer, an incoming ticket
     10–19   the answer is buried → the tickets link into a graph
     19–24   the promise: "every closed ticket answers the next one"
     24–40   section · how the graph grows: mask → extract → nodes → growth
     40–60   section · how a search runs: query → four channels → fused
             ranking → cited reply → two entry points
     60–76   section · why it is hard: two columns, six problems, six fixes
     76–88   numbers, as bar charts and two tiles
     88–100  three lines, big serif type (the last one is the call to act)
     100–110 one more thing: 14 tickets merged into one troubleshooting doc
     110–120 the bento, held

   Provenance of on-screen strings:
     #2948 subject / error lines / claim / quote / comment id ... real closed
       ticket from the 2.0 extraction output (customer data removed); the
       masked name and e-mail are invented placeholders
     #3256 subject + link ......................... real closed ticket
     the "new ticket" card ........................ slide-00 ticket, de-identified
     graph layout, other ticket ids, channel hits,
       fused-bar widths ............................ illustrative
     "3,400+" .................................... user-directed round figure
     comparison numbers ........................... user's card, verbatim
     "one more thing" ............................. Crystal 开题报告 v2 (group
       B003: 14 tickets, 28/28 citations verified; steps abridged) */
import { createElement as h, useState } from 'react'
import { Easing as E, Sprite, Stage, useTime } from './engine'
// palette, fonts and time helpers are shared with the art module (art.jsx)
import {
  INK,
  MUT,
  SUB,
  FAINT,
  HAIR,
  WASH,
  EDGE,
  ACCENT,
  TONE,
  SANS,
  SERIF,
  MONO,
  SH_CARD,
  SH_SOFT,
  seg,
  cycle,
  bell,
  S,
  cubicAt,
  Engineer,
  Icon,
  CheckCircle,
  Checkbox,
  TeamVignette,
  RepeatVignette,
  AskVignette,
  MiniTicket,
  DocGather,
} from './art'
import { G as BASE, clamp01, fIdx, lerp, mulberry32, prefersReducedMotion } from '../shared'

// ---------- helpers ----------
// fade in over dIn after tIn, fade out over dOut before tOut
const win = (t, tIn, tOut, dIn = 0.5, dOut = 0.4) =>
  seg(t, tIn, tIn + dIn, E.easeOutCubic) * (1 - seg(t, tOut - dOut, tOut, E.easeInOutQuad))
const rise = (p, dy) => ({
  opacity: clamp01(p),
  transform: `translateY(${(1 - p) * (dy || 12)}px)`,
})
const popStyle = (p, origin) => ({
  display: 'inline-block',
  opacity: clamp01(p * 1.4),
  transform: `scale(${Math.max(0.01, p)})`,
  transformOrigin: origin || '50% 50%',
})
// `key` may ride along in the style object; it belongs on the element
const abs = ({ key, ...style }, ...kids) =>
  h('div', { key: key, style: Object.assign({ position: 'absolute' }, style) }, ...kids)
const sans = (size, color, extra) =>
  Object.assign({ fontFamily: SANS, fontSize: size, color: color || INK }, extra)
const serif = (size, color, extra) =>
  Object.assign({ fontFamily: SERIF, fontSize: size, color: color || INK }, extra)
const mono = (size, color, extra) =>
  Object.assign({ fontFamily: MONO, fontSize: size, color: color || INK }, extra)
const card = ({ key, ...style }, ...kids) =>
  h(
    'div',
    {
      key: key,
      style: Object.assign(
        {
          position: 'absolute',
          boxSizing: 'border-box',
          background: '#fff',
          borderRadius: 10,
          boxShadow: SH_CARD,
        },
        style
      ),
    },
    ...kids
  )
const tag = (text, tone, opts) => {
  const o = opts || {}
  return h(
    'span',
    {
      key: o.key,
      style: {
        display: 'inline-block',
        fontFamily: o.mono ? MONO : SANS,
        fontSize: o.size || 13,
        lineHeight: 1.25,
        fontWeight: 500,
        color: tone.fg,
        background: tone.bg,
        borderRadius: 4,
        padding: '3px 7px',
        whiteSpace: 'nowrap',
        boxSizing: 'border-box',
        width: o.width,
        textAlign: o.width ? 'center' : undefined,
      },
    },
    text
  )
}
const divider = (key) =>
  h('div', { key: key, style: { height: 1, background: HAIR, margin: '14px 0' } })

// ---------- timeline ----------
const END = 120
const T = {
  ticket: 4.5,
  traits: 6.8,
  buried: 10,
  connect: 14,
  promise: 19,
  secForm: 24,
  form: 25.5,
  extract: 29,
  nodes: 33,
  grow: 36,
  secSearch: 40,
  query: 41.5,
  channels: 44,
  rank: 49.5,
  answer: 53.5,
  entry: 56.5,
  secHard: 60,
  hard: 61.5,
  solveA: 64,
  solveB: 70,
  nums: 76,
  nums2: 81.5,
  values: 88,
  crystal: 100,
  crystalDoc: 101.5,
  bento: 110,
}

// ---------- real strings ----------
const OLD = {
  id: '2948',
  subj: 'dify 升级到 3.9.5 后代码执行报错',
  claim:
    'The failure occurred during Sandbox DifySeccomp initialization, not in the customer’s Python logic.',
  quote: '当前错误发生在 sandbox 初始化 DifySeccomp 阶段，不是代码逻辑本身的问题。',
  quoteId: 'comment_id=…5924',
}
const NEIGHBOR = { id: '3256', subj: 'Sandbox unable to run in v3.9.5, …' }

// =====================================================================
// THE GRAPH — built once, deterministically. World space 1000 × 560.
// Tickets are ink dots; features are coloured: keyword (green),
// version (blue), link (purple). Five loose clusters, a few bridges.
// =====================================================================
const KIND_TONE = { kw: TONE.green, ver: TONE.blue, link: TONE.purple }

// Same seed and construction as the post's graph figures (../shared G): the
// world layout, the sea grid and the edges come from there; the film adds
// growth order and search hits on its own copies of tickets and features.
const G = (() => {
  const { W, H, edges, byFeat } = BASE
  const tickets = BASE.tickets.map((tk) => Object.assign({}, tk))
  const feats = BASE.feats.map((f) => Object.assign({}, f))

  // growth order (act 3): breadth-first from #2948 through shared features
  const dist = tickets.map(() => 99)
  dist[0] = 0
  const queue = [0]
  while (queue.length) {
    const i = queue.shift()
    tickets[i].fs.forEach((f) =>
      byFeat[f].forEach((j) => {
        if (dist[j] === 99) {
          dist[j] = dist[i] + 1
          queue.push(j)
        }
      })
    )
  }
  const t0 = tickets[0]
  const rest = tickets
    .map((_, i) => i)
    .filter((i) => i !== 0)
    .sort(
      (a, b) =>
        dist[a] - dist[b] ||
        Math.hypot(tickets[a].x - t0.x, tickets[a].y - t0.y) -
          Math.hypot(tickets[b].x - t0.x, tickets[b].y - t0.y)
    )
  t0.growT = T.nodes + 0.2
  rest.forEach((i, k) => {
    tickets[i].growT = T.grow + 0.15 + 3.0 * Math.pow(k / (rest.length - 1), 0.85)
  })
  feats.forEach((f, j) => {
    f.growT = t0.fs.includes(j)
      ? T.nodes + 0.95
      : Math.min(...byFeat[j].map((i) => tickets[i].growT)) - 0.12
  })

  // search (act 4): four channels, each with its own hit times
  const hits = tickets.map(() => [])
  const hit = (i, ch, t) => {
    if (!hits[i].some((x) => x.ch === ch)) hits[i].push({ ch: ch, t: t })
  }
  tickets.forEach((tk, i) => {
    if (tk.c === 0) {
      hit(i, 0, T.channels + 0.25 + 1.2 * clamp01(Math.hypot(tk.x - t0.x, tk.y - t0.y) / 230))
    }
  })
  const txtOther = tickets.findIndex(
    (tk, i) => i > 1 && tk.c === 0 && tk.fs.includes(fIdx('seccomp'))
  )
  hit(0, 1, T.channels + 1.6)
  if (txtOther > 0) hit(txtOther, 1, T.channels + 1.75)
  const KWF = [fIdx('seccomp'), fIdx('3.9.x')]
  const pulses = []
  edges.forEach((e, k) => {
    if (!KWF.includes(e.f)) return
    const s = T.channels + 2.95 + (k % 7) * 0.05
    pulses.push({ from: 'f', f: e.f, t: e.t, t0: s, t1: s + 0.45, ch: 2 })
    hit(e.t, 2, s + 0.45)
  })
  t0.fs.forEach((f) =>
    pulses.push({ from: 't', f: f, t: 0, t0: T.channels + 3.6, t1: T.channels + 3.95, ch: 3 })
  )
  hit(0, 3, T.channels + 3.6)
  tickets.forEach((tk, i) => {
    if (i === 0) return
    const shared = tk.fs.filter((f) => t0.fs.includes(f))
    if (shared.length < 2) return
    shared.forEach((f, k) =>
      pulses.push({
        from: 'f',
        f: f,
        t: i,
        t0: T.channels + 4.0 + k * 0.08,
        t1: T.channels + 4.4 + k * 0.08,
        ch: 3,
      })
    )
    hit(i, 3, T.channels + 4.45)
  })
  const score = (i) => hits[i].length + (i === 0 ? 2 : i === 1 ? 1 : 0) - i * 1e-4
  const top = tickets
    .map((_, i) => i)
    .sort((a, b) => score(b) - score(a))
    .slice(0, 5)
  return {
    W: W,
    H: H,
    tickets: tickets,
    feats: feats,
    edges: edges,
    hits: hits,
    pulses: pulses,
    top: top,
  }
})()

const CH = [
  { name: '语义', desc: '意思相近的旧工单', icon: 'ripple', tone: TONE.ink, t: T.channels + 0.2 },
  { name: '原文', desc: '报错串一字不差', icon: 'text', tone: TONE.gray, t: T.channels + 1.3 },
  {
    name: '关键词',
    desc: '关键词、版本号相同',
    icon: 'tag',
    tone: TONE.green,
    t: T.channels + 2.4,
  },
  { name: '图关系', desc: '在图里彼此相邻', icon: 'nodes', tone: TONE.blue, t: T.channels + 3.5 },
]

// cameras: world → screen
const T0 = G.tickets[0]
const CAM1 = { cx: 500, cy: 280, s: 0.92, ox: 640, oy: 342 }
const CAM3A = { cx: T0.x, cy: T0.y, s: 1.9, ox: 900, oy: 320 }
const CAM3B = { cx: 500, cy: 280, s: 0.6, ox: 915, oy: 345 }
const CAM4 = { cx: 500, cy: 280, s: 0.68, ox: 392, oy: 366 }
const mixCam = (a, b, p) => ({
  cx: lerp(a.cx, b.cx, p),
  cy: lerp(a.cy, b.cy, p),
  s: lerp(a.s, b.s, p),
  ox: lerp(a.ox, b.ox, p),
  oy: lerp(a.oy, b.oy, p),
})
const toScreen = (cam, x, y) => [cam.ox + (x - cam.cx) * cam.s, cam.oy + (y - cam.cy) * cam.s]

// query chips (act 4) — the keyword channel's dashed lines start here
const QCHIPS = [
  { key: 'seccomp', f: fIdx('seccomp'), x: 136, w: 80, tone: TONE.green },
  { key: '3.9.x', f: fIdx('3.9.x'), x: 226, w: 60, tone: TONE.blue },
]
const QCHIP_Y = 128

const haloText = (key, x, y, text, size, color, extra) =>
  h(
    'text',
    Object.assign(
      {
        key: key,
        x: x,
        y: y,
        fontFamily: SANS,
        fontSize: size,
        fill: color,
        fontWeight: 500,
        stroke: '#fff',
        strokeWidth: 4,
        strokeLinejoin: 'round',
        paintOrder: 'stroke',
      },
      extra
    ),
    text
  )

function GraphLayer() {
  const t = useTime()
  let act = 0,
    op = 0
  if (t >= T.buried && t < T.promise + 0.6) {
    act = 1
    op = win(t, T.buried, T.promise + 0.5, 0.3, 0.6)
  } else if (t >= T.nodes - 0.2 && t < T.secSearch + 0.3) {
    act = 3
    op = win(t, T.nodes - 0.2, T.secSearch + 0.2, 0.3, 0.3)
  } else if (t >= T.query - 0.3 && t < T.secHard + 0.4) {
    act = 4
    op = win(t, T.query - 0.3, T.secHard + 0.3, 0.5, 0.4)
  }
  if (op <= 0) return null
  const cam =
    act === 1
      ? CAM1
      : act === 3
      ? mixCam(CAM3A, CAM3B, seg(t, T.grow - 0.1, T.grow + 3.3, E.easeInOutCubic))
      : CAM4
  const tr = 2.3 * cam.s + 1.5,
    fr = 3.4 * cam.s + 2.4
  const rankP = act === 4 ? seg(t, T.rank + 0.1, T.rank + 0.8) : 0
  const quietP = act === 4 ? seg(t, T.answer, T.answer + 0.6) : 0
  const isTop = (i) => G.top.includes(i)

  const tp = G.tickets.map((tk, i) => {
    let x = tk.x,
      y = tk.y,
      show = 1,
      fillOp = 0.78,
      fill = INK,
      dim = 1
    if (act === 1) {
      const m = seg(
        t,
        T.connect + 0.3 + tk.d2 * 0.7,
        T.connect + 1.9 + tk.d2 * 0.7,
        E.easeInOutCubic
      )
      x = lerp(tk.x0, tk.x, m)
      y = lerp(tk.y0, tk.y, m)
      show = seg(t, T.buried + 0.1 + tk.d1 * 1.1, T.buried + 0.45 + tk.d1 * 1.1, E.easeOutCubic)
      fillOp = lerp(0.26, 0.78, m)
    } else if (act === 3) {
      show = seg(t, tk.growT, tk.growT + 0.4, E.easeOutBack)
      if (i === 0) fill = ACCENT
    } else {
      const got = G.hits[i].some((x) => x.t <= t)
      fill = i === 0 && rankP > 0.5 ? ACCENT : got ? INK : FAINT
      fillOp = got ? 0.9 : 0.8
      dim = lerp(1, isTop(i) ? 1 : 0.22, rankP) * lerp(1, i === 0 ? 1 : 0.45, quietP)
    }
    const p = toScreen(cam, x, y)
    return { x: p[0], y: p[1], show: show, fill: fill, fillOp: fillOp, dim: dim }
  })
  const fp = G.feats.map((f, j) => {
    let show = 1
    if (act === 1)
      show = seg(t, T.connect + 0.9 + j * 0.05, T.connect + 1.4 + j * 0.05, E.easeOutBack)
    else if (act === 3) show = seg(t, f.growT, f.growT + 0.4, E.easeOutBack)
    const p = toScreen(cam, f.x, f.y)
    const dim =
      act === 4 ? lerp(1, G.tickets[0].fs.includes(j) ? 1 : 0.3, rankP) * lerp(1, 0.45, quietP) : 1
    return { x: p[0], y: p[1], show: show, dim: dim }
  })

  const els = []
  // semantic ripple
  if (act === 4) {
    const p = seg(t, CH[0].t, CH[0].t + 1.3, E.easeOutCubic)
    if (p > 0 && p < 1) {
      els.push(
        h('circle', {
          key: 'ripple',
          cx: tp[0].x,
          cy: tp[0].y,
          r: 230 * cam.s * p,
          fill: TONE.blue.fg,
          fillOpacity: 0.05 * (1 - p),
          stroke: TONE.blue.fg,
          strokeOpacity: 0.5 * (1 - p),
          strokeWidth: 1.5,
        })
      )
    }
  }
  // edges
  const tint = {}
  if (act === 4)
    G.pulses.forEach((pl) => {
      if (t >= pl.t0) tint[pl.t + ':' + pl.f] = CH[pl.ch].tone.fg
    })
  G.edges.forEach((e, k) => {
    const a = tp[e.t],
      b = fp[e.f]
    let p = 1
    if (act === 1)
      p = seg(
        t,
        T.connect + 1.3 + G.tickets[e.t].d2 * 0.9,
        T.connect + 2.2 + G.tickets[e.t].d2 * 0.9
      )
    else if (act === 3) {
      const s = Math.max(G.tickets[e.t].growT, G.feats[e.f].growT)
      p = seg(t, s + 0.1, s + 0.55)
    }
    if (p <= 0) return
    const tc = tint[e.t + ':' + e.f]
    const eo = act === 4 ? Math.min(a.dim, b.dim) * (tc ? 0.75 : 1) : 1
    const len = Math.hypot(b.x - a.x, b.y - a.y)
    els.push(
      h('line', {
        key: 'e' + k,
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        stroke: tc || EDGE,
        strokeWidth: tc ? 1.4 : 1,
        strokeOpacity: eo,
        strokeDasharray: len,
        strokeDashoffset: len * (1 - p),
      })
    )
  })
  // keyword channel: dashed lines from the query chips to their features
  if (act === 4) {
    const lp = seg(t, CH[2].t + 0.1, CH[2].t + 0.5, E.easeInOutCubic) * (1 - rankP)
    if (lp > 0)
      QCHIPS.forEach((c) => {
        const x1 = c.x + c.w / 2,
          y1 = QCHIP_Y + 24,
          b = fp[c.f]
        els.push(
          h('line', {
            key: 'q' + c.key,
            x1: x1,
            y1: y1,
            x2: lerp(x1, b.x, lp),
            y2: lerp(y1, b.y, lp),
            stroke: TONE.green.fg,
            strokeWidth: 1.3,
            strokeDasharray: '4 4',
            strokeOpacity: 0.8,
          })
        )
      })
    // pulses travelling along edges
    G.pulses.forEach((pl, k) => {
      const p = seg(t, pl.t0, pl.t1, E.easeInOutQuad)
      if (p <= 0 || p >= 1) return
      const a = pl.from === 'f' ? fp[pl.f] : tp[pl.t],
        b = pl.from === 'f' ? tp[pl.t] : fp[pl.f]
      els.push(
        h('circle', {
          key: 'p' + k,
          cx: lerp(a.x, b.x, p),
          cy: lerp(a.y, b.y, p),
          r: 3,
          fill: CH[pl.ch].tone.fg,
        })
      )
    })
  }
  // features: solid dot on a soft tinted halo
  G.feats.forEach((f, j) => {
    const s = fp[j]
    if (s.show <= 0) return
    const tone = KIND_TONE[f.kind]
    els.push(
      h('circle', {
        key: 'fh' + j,
        cx: s.x,
        cy: s.y,
        r: (fr + 4.5) * Math.max(0.01, s.show),
        fill: tone.bg,
        opacity: s.dim,
      })
    )
    els.push(
      h('circle', {
        key: 'f' + j,
        cx: s.x,
        cy: s.y,
        r: fr * Math.max(0.01, s.show),
        fill: tone.fg,
        stroke: '#fff',
        strokeWidth: 1.6,
        opacity: s.dim,
      })
    )
  })
  // tickets + channel rings
  G.tickets.forEach((tk, i) => {
    const s = tp[i]
    if (s.show <= 0) return
    els.push(
      h('circle', {
        key: 't' + i,
        cx: s.x,
        cy: s.y,
        r: tr * Math.max(0.01, s.show),
        fill: s.fill,
        fillOpacity: s.fillOp,
        opacity: s.dim,
      })
    )
    if (act === 4)
      G.hits[i].forEach((hh, k) => {
        const p = seg(t, hh.t, hh.t + 0.35, E.easeOutCubic)
        if (p <= 0) return
        const r = tr + 2.6 * (k + 1)
        els.push(
          h('circle', {
            key: 'r' + i + '-' + k,
            cx: s.x,
            cy: s.y,
            r: lerp(r + 7, r, p),
            fill: 'none',
            stroke: CH[hh.ch].tone.fg,
            strokeWidth: 1.4,
            strokeOpacity: p * s.dim,
          })
        )
      })
  })
  // act 1: the buried ticket — a hollow ring with a slow sonar ping
  if (act === 1) {
    const rp = win(t, T.buried + 0.9, T.connect + 0.5, 0.4, 0.4)
    const ping = cycle(t - T.buried, 1.3)
    if (rp > 0)
      els.push(
        h('circle', {
          key: 'bur',
          cx: tp[0].x,
          cy: tp[0].y,
          r: 9,
          fill: 'none',
          stroke: INK,
          strokeWidth: 1.6,
          opacity: rp,
        }),
        h('circle', {
          key: 'burp',
          cx: tp[0].x,
          cy: tp[0].y,
          r: 9 + 16 * ping,
          fill: 'none',
          stroke: INK,
          strokeWidth: 1,
          opacity: rp * 0.5 * (1 - ping),
        })
      )
  }
  // labels
  const lsize = 10.5 + 1.8 * cam.s
  const labOp = act === 1 ? seg(t, T.connect + 2.3, T.connect + 2.9) : 1
  G.feats.forEach((f, j) => {
    const s = fp[j]
    const o = labOp * clamp01(s.show) * s.dim
    if (o <= 0) return
    els.push(
      haloText('l' + j, s.x + fr + 5, s.y + lsize * 0.36, f.key, lsize, KIND_TONE[f.kind].fg, {
        opacity: o,
      })
    )
  })
  if (act === 3) {
    const o = seg(t, T.nodes + 0.4, T.nodes + 0.8)
    els.push(
      haloText('t0l', tp[0].x - tr - 6, tp[0].y + 5, '#' + OLD.id, lsize + 1, INK, {
        opacity: o,
        textAnchor: 'end',
        fontFamily: MONO,
        fontWeight: 600,
      })
    )
  }
  if (act === 4) {
    const o = seg(t, T.rank + 0.6, T.rank + 1.0, E.easeOutBack)
    if (o > 0)
      els.push(
        h(
          'g',
          {
            key: 'win',
            opacity: clamp01(o),
            transform: `translate(${tp[0].x - 16} ${tp[0].y - 16}) scale(${Math.max(0.01, o)})`,
          },
          h('rect', { x: -78, y: -14, width: 74, height: 24, rx: 5, fill: TONE.blue.bg }),
          h(
            'text',
            {
              x: -41,
              y: 3,
              textAnchor: 'middle',
              fontFamily: MONO,
              fontSize: 12.5,
              fontWeight: 600,
              fill: TONE.blue.fg,
            },
            '#' + OLD.id
          )
        )
      )
  }
  // act 3 shares the stage with the ticket page on the left: fade the graph out before it
  const body =
    act === 3
      ? [
          h(
            'defs',
            { key: 'defs' },
            h(
              'linearGradient',
              { id: 'g3fade', gradientUnits: 'userSpaceOnUse', x1: 570, y1: 0, x2: 650, y2: 0 },
              h('stop', { offset: 0, stopColor: '#000' }),
              h('stop', { offset: 1, stopColor: '#fff' })
            ),
            h(
              'mask',
              { id: 'g3mask', maskUnits: 'userSpaceOnUse', x: 0, y: 0, width: 1280, height: 720 },
              h('rect', { x: 0, y: 0, width: 1280, height: 720, fill: 'url(#g3fade)' })
            )
          ),
          h('g', { key: 'g', mask: 'url(#g3mask)' }, els),
        ]
      : els
  return h(
    'svg',
    {
      width: 1280,
      height: 720,
      style: {
        position: 'absolute',
        left: 0,
        top: 0,
        opacity: op,
        overflow: 'visible',
        pointerEvents: 'none',
      },
    },
    body
  )
}

// =====================================================================
// ACT 0 — the support desk (0–10)
// =====================================================================
// help request → new ticket: a single curve, with a spark running along it
const LINK = [
  [560, 160],
  [616, 160],
  [604, 278],
  [652, 278],
]
function SupportIntro() {
  const t = useTime()
  const o = 1 - seg(t, T.buried - 0.6, T.buried, E.easeInOutQuad)
  if (o <= 0) return null
  const title = 1 - seg(t, T.ticket - 0.5, T.ticket, E.easeInOutQuad)
  const notice = seg(t, 1.5, 2.1, E.easeOutCubic)
  const link = seg(t, T.ticket, T.ticket + 0.6, E.easeInOutCubic)
  const run = cycle(Math.max(0, t - T.ticket - 0.6), 1.8)
  const pt = (s) => [
    cubicAt(LINK[0][0], LINK[1][0], LINK[2][0], LINK[3][0], s),
    cubicAt(LINK[0][1], LINK[1][1], LINK[2][1], LINK[3][1], s),
  ]
  const spark = pt(E.easeInOutQuad(run))
  return abs(
    { inset: 0, opacity: o },
    abs({ left: 100, top: 156 }, h(Engineer, { t: t, pose: 'typing' })),
    abs(
      Object.assign({ left: 660, top: 250 }, rise(title * seg(t, 0.2, 0.9, E.easeOutCubic), 10)),
      h(
        'div',
        { style: serif(46, INK, { fontWeight: 700, letterSpacing: '-0.01em', lineHeight: 1.35 }) },
        '售后工程师的一天，'
      ),
      h(
        'div',
        { style: serif(46, INK, { fontWeight: 700, letterSpacing: '-0.01em', lineHeight: 1.35 }) },
        '从一条求助开始。'
      )
    ),
    card(
      Object.assign(
        {
          left: 360,
          top: 138,
          padding: '10px 16px 10px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          borderRadius: 8,
        },
        rise(notice, 10)
      ),
      h(Icon, {
        name: 'bell',
        size: 20,
        color: INK,
        accent: TONE.pink.fg,
        p: seg(t, 1.5, 2.2),
        t: Math.max(0, t - 1.9),
        sw: 1.6,
      }),
      h('span', { style: sans(15, INK, { fontWeight: 500 }) }, '客户发来一条求助')
    ),
    h(
      'svg',
      {
        width: 1280,
        height: 720,
        style: { position: 'absolute', left: 0, top: 0, pointerEvents: 'none' },
      },
      h('path', {
        d: `M${LINK[0]}C${LINK[1]} ${LINK[2]} ${LINK[3]}`,
        fill: 'none',
        stroke: FAINT,
        strokeWidth: 1.5,
        strokeLinecap: 'round',
        pathLength: 1,
        strokeDasharray: 1,
        strokeDashoffset: 1 - link,
        opacity: link > 0 ? 1 : 0,
      }),
      link >= 1
        ? h('circle', { cx: spark[0], cy: spark[1], r: 3, fill: TONE.pink.fg, opacity: bell(run) })
        : null,
      h('circle', { cx: LINK[3][0], cy: LINK[3][1], r: 3 * link, fill: FAINT })
    )
  )
}

const propRow = (label, value, key, style) =>
  h(
    'div',
    {
      key: key,
      style: Object.assign({ display: 'flex', alignItems: 'center', minHeight: 30 }, style),
    },
    h('span', { style: sans(13.5, SUB, { width: 78, flexShrink: 0 }) }, label),
    h('div', { style: { display: 'flex', gap: 6, alignItems: 'center' } }, value)
  )

function NewTicket() {
  const t = useTime()
  const o = win(t, T.ticket + 0.2, T.buried, 0.6, 0.6)
  if (o <= 0) return null
  const inn = seg(t, T.ticket + 0.2, T.ticket + 0.9, E.easeOutCubic)
  const body = seg(t, T.ticket + 0.9, T.ticket + 1.5, E.easeOutCubic)
  const trait = (k) => seg(t, T.traits + k * 0.45, T.traits + 0.45 + k * 0.45, E.easeOutBack)
  const TRAITS = [
    ['重复出现', TONE.ink],
    ['版本相关', TONE.blue],
    ['要翻源码', TONE.purple],
  ]
  return card(
    {
      left: 660,
      top: 146,
      width: 520,
      padding: '22px 28px 24px',
      opacity: o,
      transform: `translateY(${(1 - inn) * 14}px)`,
    },
    h('div', { style: sans(13, SUB) }, '新工单 · 刚刚'),
    h(
      'div',
      {
        style: sans(25, INK, {
          fontWeight: 700,
          letterSpacing: '-0.01em',
          marginTop: 8,
          marginBottom: 12,
        }),
      },
      'K8s 部署，3.8.0 升级到 3.9.8'
    ),
    propRow('状态', tag('新建', TONE.pink), 's'),
    propRow('版本', h('span', { style: sans(14, INK) }, '3.8.0 → 3.9.8'), 'v'),
    propRow(
      '特点',
      TRAITS.map(([txt, tone], k) =>
        h('span', { key: txt, style: popStyle(trait(k), '0 50%') }, tag(txt, tone))
      ),
      'x'
    ),
    divider('d'),
    h(
      'div',
      { style: rise(body, 8) },
      h('div', { style: sans(15, INK, { marginBottom: 10 }) }, '升级后，代码节点直接崩：'),
      h(
        'div',
        { style: { background: WASH, borderRadius: 6, padding: '12px 14px' } },
        [
          'process exited with code -1',
          'panic: could not create filter goroutine 17',
          'main.DifySeccomp(…)',
        ].map((l) => h('div', { key: l, style: mono(12.5, INK, { lineHeight: '20px' }) }, l))
      )
    )
  )
}

// =====================================================================
// ACT 1 — buried (10–14), the graph forms (14–19), the promise (19–24)
// =====================================================================
function BuriedTag() {
  const t = useTime()
  const o = win(t, T.buried + 1.2, T.connect + 0.3, 0.4, 0.4)
  if (o <= 0) return null
  const p = toScreen(CAM1, T0.x0, T0.y0)
  return abs(
    { left: 0, top: 0, width: 1280, height: 720, opacity: o, pointerEvents: 'none' },
    h(
      'svg',
      { width: 1280, height: 720, style: { position: 'absolute', left: 0, top: 0 } },
      h('line', {
        x1: p[0] + 7,
        y1: p[1] - 7,
        x2: p[0] + 28,
        y2: p[1] - 30,
        stroke: FAINT,
        strokeWidth: 1.2,
      })
    ),
    card(
      {
        left: p[0] + 30,
        top: p[1] - 70,
        padding: '10px 14px',
        borderRadius: 8,
        transform: `translateY(${(1 - o) * 6}px)`,
      },
      h(
        'div',
        { style: { display: 'flex', gap: 8, alignItems: 'center' } },
        h('span', { style: mono(12.5, INK, { fontWeight: 600 }) }, '#' + OLD.id),
        tag('已关闭', TONE.gray, { size: 11.5 }),
        h('span', { style: sans(12, SUB) }, '半年前')
      ),
      h('div', { style: sans(13, MUT, { marginTop: 5, whiteSpace: 'nowrap' }) }, OLD.subj)
    )
  )
}

function CountUp() {
  const t = useTime()
  const o = win(t, T.connect + 0.1, T.promise, 0.5, 0.5)
  if (o <= 0) return null
  const n = Math.round(3400 * seg(t, T.connect + 0.2, T.connect + 2.2, E.easeOutCubic))
  const plus = seg(t, T.connect + 2.2, T.connect + 2.5)
  return abs(
    { left: 80, top: 34, opacity: o, display: 'flex', alignItems: 'baseline', gap: 12 },
    h(
      'span',
      {
        style: sans(44, INK, {
          fontWeight: 700,
          letterSpacing: '-0.02em',
          fontVariantNumeric: 'tabular-nums',
        }),
      },
      n.toLocaleString('en-US'),
      h('span', { style: { opacity: plus } }, '+')
    ),
    h('span', { style: sans(16, MUT) }, '张已关闭工单')
  )
}

function PromiseScene() {
  const t = useTime()
  const o = win(t, T.promise, T.secForm, 0.6, 0.5)
  if (o <= 0) return null
  const line = (k) => seg(t, T.promise + 0.4 + k * 0.35, T.promise + 1.0 + k * 0.35, E.easeOutCubic)
  const res = seg(t, T.promise + 1.4, T.promise + 2.0, E.easeOutCubic)
  return abs(
    { inset: 0, opacity: o },
    abs({ left: 90, top: 176 }, h(Engineer, { t: t - T.promise, pose: 'relaxed' })),
    abs({ left: 128, top: 134 }, h('div', { style: sans(18, MUT) }, '这次，不用从头翻了。')),
    abs(
      { left: 650, top: 150, width: 560 },
      h(
        'div',
        {
          style: Object.assign(
            sans(15, SUB, { marginBottom: 18, display: 'flex', alignItems: 'center', gap: 8 }),
            rise(line(0), 8)
          ),
        },
        h(Icon, {
          name: 'graph',
          size: 18,
          p: seg(t, T.promise + 0.4, T.promise + 1.3),
          t: t,
          sw: 1.8,
        }),
        '历史工单知识库'
      ),
      h(
        'div',
        {
          style: Object.assign(
            serif(42, INK, { fontWeight: 700, lineHeight: 1.35 }),
            rise(line(1), 10)
          ),
        },
        '每一张关闭的工单，'
      ),
      h(
        'div',
        {
          style: Object.assign(
            serif(42, INK, { fontWeight: 700, lineHeight: 1.35 }),
            rise(line(2), 10)
          ),
        },
        '都是下一张的答案。'
      )
    ),
    card(
      Object.assign({ left: 650, top: 360, width: 520, padding: '18px 22px' }, rise(res, 12)),
      h(
        'div',
        { style: { display: 'flex', alignItems: 'center', gap: 10 } },
        h(CheckCircle, { size: 20, color: ACCENT, p: seg(t, T.promise + 1.7, T.promise + 2.4) }),
        h('span', { style: sans(15, INK, { fontWeight: 600 }) }, '找到相关工单'),
        h('span', { style: mono(13, SUB, { marginLeft: 'auto' }) }, '#' + OLD.id)
      ),
      h('div', { style: sans(16, INK, { marginTop: 10 }) }, OLD.subj),
      h(
        'div',
        { style: { marginTop: 12, borderLeft: `3px solid ${INK}`, paddingLeft: 12 } },
        h('div', { style: sans(13.5, MUT, { lineHeight: 1.6 }) }, OLD.quote)
      )
    )
  )
}

// =====================================================================
// Section titles (Notion H1 on a clean page)
// =====================================================================
// Notion page header: a line icon that draws itself above the H1
const SECTIONS = [
  { start: T.secForm, end: T.form, title: '图是怎么长出来的', icon: 'graph' },
  { start: T.secSearch, end: T.query, title: '一次检索怎么走', icon: 'search' },
  { start: T.secHard, end: T.hard, title: '工单支持，难在哪里', icon: 'peak' },
]
function SectionTitle() {
  const t = useTime()
  const s = SECTIONS.find((x) => t >= x.start && t < x.end)
  if (!s) return null
  const o = win(t, s.start, s.end, 0.25, 0.3)
  const p = seg(t, s.start + 0.05, s.start + 0.6, E.easeOutCubic)
  return abs(
    { inset: 0, background: '#fff', opacity: o },
    abs(
      { left: 140, top: 226, ...rise(p, 12) },
      h(Icon, {
        name: s.icon,
        size: 76,
        p: seg(t, s.start + 0.1, s.start + 1.0),
        t: t - s.start,
        sw: 1.15,
        style: { marginBottom: 26, marginLeft: -4 },
      }),
      h('div', { style: sans(66, INK, { fontWeight: 700, letterSpacing: '-0.02em' }) }, s.title)
    )
  )
}

// =====================================================================
// ACT 2 — how the graph grows (25.5–40)
// =====================================================================
const PG = { x: 80, y: 92, w: 470 }
function FormPage() {
  const t = useTime()
  const o = win(t, T.form, T.secSearch, 0.5, 0.4)
  if (o <= 0) return null
  const inn = seg(t, T.form, T.form + 0.6, E.easeOutCubic)
  const scanN = seg(t, T.form + 1.1, T.form + 1.7),
    swapN = seg(t, T.form + 1.7, T.form + 2.1, E.easeOutBack)
  const scanE = seg(t, T.form + 1.9, T.form + 2.5),
    swapE = seg(t, T.form + 2.5, T.form + 2.9, E.easeOutBack)
  const badge = seg(t, T.form + 2.9, T.form + 3.3, E.easeOutBack)
  const quoteHi = seg(t, T.extract + 1.4, T.extract + 1.9)
  const pii = (plain, token, scan, swap) =>
    swap > 0
      ? h(
          'span',
          { style: popStyle(swap, '0 50%') },
          tag(token, TONE.pink, { mono: true, size: 12.5 })
        )
      : h(
          'span',
          { style: sans(14, INK, { position: 'relative', display: 'inline-block' }) },
          plain,
          h('span', {
            style: {
              position: 'absolute',
              left: 0,
              bottom: -3,
              height: 2,
              width: `${scan * 100}%`,
              background: TONE.pink.fg,
            },
          })
        )
  return card(
    {
      left: PG.x,
      top: PG.y,
      width: PG.w,
      padding: '22px 24px',
      opacity: o,
      transform: `translateY(${(1 - inn) * 14}px)`,
    },
    h(
      'div',
      { style: { display: 'flex', alignItems: 'center', gap: 8 } },
      h('span', { style: mono(13, SUB) }, '#' + OLD.id),
      tag('已关闭', TONE.gray, { size: 12 }),
      h(
        'span',
        { style: Object.assign({ marginLeft: 'auto' }, popStyle(badge, '100% 50%')) },
        tag('已脱敏', TONE.pink, { size: 12 })
      )
    ),
    h(
      'div',
      { style: sans(22, INK, { fontWeight: 700, marginTop: 10, marginBottom: 10 }) },
      OLD.subj
    ),
    propRow('提交人', pii('王小明', '[NAME_1]', scanN, swapN), 'n'),
    propRow('邮箱', pii('xiaoming@example.com', '[EMAIL_1]', scanE, swapE), 'e'),
    propRow('版本', h('span', { style: sans(14, INK) }, '3.9.5'), 'v'),
    divider('d'),
    h('div', { style: sans(12.5, SUB, { marginBottom: 6 }) }, '工程师回复'),
    h(
      'div',
      { style: sans(15, INK, { lineHeight: 1.7 }) },
      h(
        'span',
        {
          style: {
            background: `rgba(230,241,250,${quoteHi})`,
            boxShadow: quoteHi > 0 ? `0 2px 0 rgba(0,112,201,${quoteHi})` : 'none',
            borderRadius: 2,
          },
        },
        OLD.quote
      )
    ),
    h(
      'div',
      { style: { background: WASH, borderRadius: 6, padding: '10px 14px', marginTop: 12 } },
      ['process exited with code -1', '[running]: main.DifySeccomp(…)'].map((l) =>
        h('div', { key: l, style: mono(12.5, INK, { lineHeight: '20px' }) }, l)
      )
    )
  )
}

const XCHIPS = [
  { key: 'sandbox', f: fIdx('sandbox'), x: 716, w: 84, tone: TONE.green },
  { key: 'seccomp', f: fIdx('seccomp'), x: 808, w: 84, tone: TONE.green },
  { key: '3.9.x', f: fIdx('3.9.x'), x: 980, w: 62, tone: TONE.blue },
]
const XCHIP_Y = 316
function ExtractPanel() {
  const t = useTime()
  const o = win(t, T.extract, T.nodes + 0.5, 0.5, 0.5)
  if (o <= 0) return null
  const s = T.extract
  const head = seg(t, s + 0.1, s + 0.5, E.easeOutCubic)
  const claim = seg(t, s + 0.4, s + 1.0, E.easeOutCubic)
  const link = seg(t, s + 1.3, s + 2.1, E.easeInOutCubic)
  const src = seg(t, s + 2.0, s + 2.4, E.easeOutCubic)
  const ok = seg(t, s + 2.3, s + 2.8, E.easeOutBack)
  const rowIn = seg(t, s + 2.4, s + 2.8, E.easeOutCubic)
  const chip = (k) => seg(t, s + 2.6 + k * 0.2, s + 3.0 + k * 0.2, E.easeOutBack)
  const flying = t >= T.nodes
  return abs(
    { inset: 0, opacity: o },
    h(
      'svg',
      {
        width: 1280,
        height: 720,
        style: { position: 'absolute', left: 0, top: 0, pointerEvents: 'none' },
      },
      h('path', {
        d: 'M634 200C592 200 584 352 540 352',
        fill: 'none',
        stroke: TONE.blue.fg,
        strokeWidth: 1.5,
        pathLength: 1,
        strokeDasharray: 1,
        strokeDashoffset: 1 - link,
      }),
      h('circle', { cx: 540, cy: 352, r: 3.2 * link, fill: TONE.blue.fg })
    ),
    abs(
      Object.assign(
        { left: 640, top: 96, display: 'flex', alignItems: 'center', gap: 8 },
        rise(head, 8)
      ),
      h(Icon, { name: 'sparkle', size: 18, accent: ACCENT, p: seg(t, s + 0.1, s + 0.8), t: t }),
      h('span', { style: sans(14, SUB) }, 'AI 提炼（只看脱敏后的文本）')
    ),
    abs(
      Object.assign(
        {
          left: 640,
          top: 128,
          width: 540,
          boxSizing: 'border-box',
          background: WASH,
          borderRadius: 8,
          padding: '14px 18px',
        },
        rise(claim, 10)
      ),
      h('div', { style: sans(12.5, SUB, { marginBottom: 6 }) }, '结论'),
      h('div', { style: sans(16, INK, { lineHeight: 1.55 }) }, OLD.claim)
    ),
    abs(
      Object.assign(
        { left: 640, top: 262, display: 'flex', alignItems: 'center', gap: 10 },
        rise(src, 8)
      ),
      h('span', { style: sans(14, MUT) }, '↳ 原话出处'),
      h('span', { style: mono(13, INK) }, '#' + OLD.id + ' · ' + OLD.quoteId),
      h(CheckCircle, { size: 18, color: TONE.blue.fg, p: seg(t, s + 2.3, s + 3.0) })
    ),
    abs(
      Object.assign({ left: 640, top: XCHIP_Y + 3 }, rise(rowIn, 6)),
      h('span', { style: sans(14, SUB) }, '关键词')
    ),
    abs(
      Object.assign({ left: 930, top: XCHIP_Y + 3 }, rise(rowIn, 6)),
      h('span', { style: sans(14, SUB) }, '版本')
    ),
    flying
      ? null
      : XCHIPS.map((c, k) =>
          abs(
            { key: c.key, left: c.x, top: XCHIP_Y },
            h(
              'span',
              { style: popStyle(chip(k), '0 50%') },
              tag(c.key, c.tone, { mono: true, size: 13, width: c.w })
            )
          )
        )
  )
}
// the extracted chips fly into the graph and become its first nodes
function FlyingChips() {
  const t = useTime()
  if (t < T.nodes || t > T.nodes + 1.3) return null
  const els = XCHIPS.map((c, k) => {
    const p = seg(t, T.nodes + k * 0.08, T.nodes + 0.95 + k * 0.08, E.easeInOutCubic)
    const dst = toScreen(CAM3A, G.feats[c.f].x, G.feats[c.f].y)
    const x0 = c.x + c.w / 2,
      y0 = XCHIP_Y + 12
    return abs(
      {
        key: c.key,
        left: lerp(x0, dst[0], p),
        top: lerp(y0, dst[1], p),
        opacity: 1 - seg(p, 0.8, 1),
        transform: `translate(-50%, -50%) scale(${lerp(1, 0.25, p)})`,
      },
      tag(c.key, c.tone, { mono: true, size: 13, width: c.w })
    )
  })
  const d = seg(t, T.nodes, T.nodes + 0.8, E.easeInOutCubic)
  const src = [PG.x + 60, PG.y + 60]
  els.push(
    abs({
      key: 'dot',
      left: lerp(src[0], CAM3A.ox, d) - 5,
      top: lerp(src[1], CAM3A.oy, d) - 5,
      width: 10,
      height: 10,
      borderRadius: 99,
      background: INK,
      opacity: 1 - seg(d, 0.85, 1),
    })
  )
  return abs({ inset: 0, pointerEvents: 'none' }, els)
}

// =====================================================================
// ACT 3 — how a search runs (41.5–60)
// =====================================================================
const QUERY = '3.9.8 升级后 panic: could not create filter · main.DifySeccomp(…)'
function QueryBox() {
  const t = useTime()
  const o = win(t, T.query, T.secHard, 0.5, 0.4)
  if (o <= 0) return null
  const n = Math.floor(QUERY.length * seg(t, T.query + 0.3, T.query + 1.8))
  const typed = QUERY.slice(0, n)
  const hi = seg(t, CH[1].t, CH[1].t + 0.3) * (1 - seg(t, T.rank, T.rank + 0.5))
  const k = QUERY.indexOf('DifySeccomp')
  const text =
    n > k + 11 && hi > 0
      ? [
          typed.slice(0, k),
          h(
            'span',
            {
              key: 'h',
              style: {
                background: `rgba(253,236,244,${hi})`,
                color: hi > 0.5 ? TONE.pink.fg : INK,
                borderRadius: 3,
              },
            },
            'DifySeccomp'
          ),
          typed.slice(k + 11),
        ]
      : typed
  const caret = t < T.query + 2.4 && Math.floor(t * 2.2) % 2 === 0
  const chip = (i) => seg(t, CH[2].t - 0.1 + i * 0.12, CH[2].t + 0.25 + i * 0.12, E.easeOutBack)
  const lab = seg(t, CH[2].t - 0.2, CH[2].t + 0.1)
  return abs(
    { inset: 0, opacity: o, pointerEvents: 'none' },
    card(
      {
        left: 60,
        top: 64,
        width: 660,
        height: 50,
        padding: '0 16px',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        borderRadius: 8,
      },
      h(Icon, {
        name: 'search',
        size: 19,
        color: SUB,
        accent: ACCENT,
        t: t,
        sw: 1.7,
        on: seg(t, T.query + 0.2, T.query + 0.5) * (1 - seg(t, T.query + 1.8, T.query + 2.3)),
      }),
      h(
        'span',
        { style: sans(15, INK, { whiteSpace: 'nowrap' }) },
        text,
        h('span', {
          style: {
            display: 'inline-block',
            width: 1.5,
            height: 18,
            marginLeft: 2,
            verticalAlign: '-3px',
            background: INK,
            opacity: caret ? 1 : 0,
          },
        })
      )
    ),
    abs(
      { left: 66, top: QCHIP_Y + 3, opacity: lab * (1 - seg(t, T.rank, T.rank + 0.5)) },
      h('span', { style: sans(13.5, SUB) }, '识别出')
    ),
    QCHIPS.map((c, i) =>
      abs(
        { key: c.key, left: c.x, top: QCHIP_Y, opacity: 1 - seg(t, T.rank, T.rank + 0.5) },
        h(
          'span',
          { style: popStyle(chip(i), '0 50%') },
          tag(c.key, c.tone, { mono: true, size: 13, width: c.w })
        )
      )
    )
  )
}

const PANEL = { x: 770, w: 440 }
function ChannelPanel() {
  const t = useTime()
  const o = win(t, T.channels - 0.1, T.rank + 0.3, 0.4, 0.4)
  if (o <= 0) return null
  return abs(
    { inset: 0, opacity: o },
    abs({ left: PANEL.x, top: 70 }, h('span', { style: sans(14, SUB) }, '四路同时找')),
    CH.map((c, k) => {
      const p = seg(t, c.t - 0.2, c.t + 0.3, E.easeOutCubic)
      const active = win(t, c.t, c.t + 1.4, 0.2, 0.4)
      const count = G.hits.filter((hs) => hs.some((x) => x.ch === k && x.t <= t)).length
      return card(
        Object.assign(
          {
            key: c.name,
            left: PANEL.x,
            top: 102 + k * 88,
            width: PANEL.w,
            height: 74,
            padding: '0 20px',
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            borderRadius: 8,
            boxShadow: SH_SOFT,
            background:
              active > 0
                ? `color-mix(in srgb, ${c.tone.bg} ${Math.round(active * 100)}%, #fff)`
                : '#fff',
          },
          rise(p, 10)
        ),
        h(
          'span',
          {
            style: {
              width: 40,
              height: 40,
              borderRadius: 9,
              flexShrink: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: `color-mix(in srgb, #fff ${Math.round(active * 100)}%, ${c.tone.bg})`,
            },
          },
          h(Icon, {
            name: c.icon,
            size: 22,
            color: c.tone.fg,
            p: seg(t, c.t - 0.2, c.t + 0.5),
            t: t,
            on: active,
            sw: 1.7,
          })
        ),
        h(
          'div',
          null,
          h('div', { style: sans(18, INK, { fontWeight: 600 }) }, c.name),
          h('div', { style: sans(13, SUB, { marginTop: 2 }) }, c.desc)
        ),
        h(
          'div',
          { style: { marginLeft: 'auto', textAlign: 'right' } },
          h(
            'div',
            { style: sans(24, c.tone.fg, { fontWeight: 600, fontVariantNumeric: 'tabular-nums' }) },
            count
          ),
          h('div', { style: sans(11.5, SUB) }, '命中')
        )
      )
    })
  )
}

function RankPanel() {
  const t = useTime()
  const o = win(t, T.rank, T.answer + 0.2, 0.4, 0.4)
  if (o <= 0) return null
  return abs(
    { inset: 0, opacity: o },
    abs(
      { left: PANEL.x, top: 70 },
      h('span', { style: sans(14, SUB) }, '合并排序：几路都找到的，排在前面')
    ),
    G.top.map((ti, k) => {
      const tk = G.tickets[ti]
      const p = seg(t, T.rank + 0.1 + k * 0.1, T.rank + 0.5 + k * 0.1, E.easeOutCubic)
      const grow = seg(t, T.rank + 0.4 + k * 0.12, T.rank + 1.3 + k * 0.12, E.easeOutCubic)
      const first = k === 0
      const hl = first ? seg(t, T.rank + 1.2, T.rank + 1.6) : 0
      const subj = ti === 0 ? OLD.subj : ti === 1 ? NEIGHBOR.subj : null
      const segs = [0, 1, 2, 3].filter((ch) => G.hits[ti].some((x) => x.ch === ch))
      return card(
        Object.assign(
          {
            key: tk.id,
            left: PANEL.x,
            top: 102 + k * 74,
            width: PANEL.w,
            height: 62,
            padding: '0 18px',
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            borderRadius: 8,
            boxShadow: SH_SOFT,
            background:
              hl > 0
                ? `color-mix(in srgb, ${TONE.blue.bg} ${Math.round(hl * 100)}%, #fff)`
                : '#fff',
          },
          rise(p, 8)
        ),
        h('span', { style: sans(18, first ? ACCENT : SUB, { fontWeight: 700, width: 14 }) }, k + 1),
        h(
          'div',
          { style: { width: 196, overflow: 'hidden' } },
          h('div', { style: mono(14, INK, { fontWeight: 600 }) }, '#' + tk.id),
          subj
            ? h(
                'div',
                {
                  style: sans(12, SUB, {
                    marginTop: 2,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }),
                },
                subj
              )
            : null
        ),
        h(
          'div',
          { style: { marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 } },
          h(
            'div',
            { style: { display: 'flex', gap: 2, width: 150 } },
            segs.map((ch) =>
              h('span', {
                key: ch,
                style: {
                  height: 10,
                  width: (ch === 0 ? 40 : ch === 2 ? 36 : 32) * grow,
                  borderRadius: 3,
                  background: CH[ch].tone.fg,
                },
              })
            )
          ),
          h(
            'span',
            { style: sans(12, SUB, { width: 28, textAlign: 'right' }) },
            segs.length + ' 路'
          )
        )
      )
    })
  )
}

function AnswerPanel() {
  const t = useTime()
  const o = win(t, T.answer, T.entry + 0.2, 0.5, 0.4)
  if (o <= 0) return null
  const s = T.answer
  const q = seg(t, s + 0.5, s + 1.0, E.easeOutCubic)
  const press = seg(t, s + 1.8, s + 2.0) * (1 - seg(t, s + 2.0, s + 2.2))
  const done = seg(t, s + 2.1, s + 2.4, E.easeOutBack)
  return card(
    {
      left: PANEL.x,
      top: 84,
      width: PANEL.w,
      padding: '20px 24px',
      opacity: o,
      transform: `translateY(${(1 - seg(t, s, s + 0.5, E.easeOutCubic)) * 12}px)`,
    },
    h(
      'div',
      { style: { display: 'flex', alignItems: 'center', gap: 8 } },
      h('span', { style: sans(13, SUB) }, '回复草稿'),
      h(
        'span',
        { style: { marginLeft: 'auto', display: 'flex', gap: 6 } },
        tag('#' + OLD.id, TONE.blue, { mono: true, size: 12 }),
        tag('#' + NEIGHBOR.id, TONE.gray, { mono: true, size: 12 })
      )
    ),
    h(
      'div',
      { style: sans(19, INK, { fontWeight: 700, lineHeight: 1.45, marginTop: 12 }) },
      '故障出在 sandbox 的 seccomp 初始化，',
      h('br'),
      '不在您的代码。'
    ),
    h(
      'div',
      {
        style: Object.assign(
          { marginTop: 14, borderLeft: `3px solid ${INK}`, paddingLeft: 14 },
          rise(q, 8)
        ),
      },
      h('div', { style: sans(14.5, MUT, { lineHeight: 1.65 }) }, OLD.quote),
      h('div', { style: mono(12, SUB, { marginTop: 6 }) }, '— #' + OLD.id + ' · ' + OLD.quoteId)
    ),
    h(
      'div',
      { style: { marginTop: 18, display: 'flex', alignItems: 'center', gap: 12 } },
      h(
        'span',
        {
          style: sans(14, '#fff', {
            fontWeight: 600,
            background: ACCENT,
            borderRadius: 6,
            padding: '8px 14px',
            transform: `scale(${1 - 0.05 * press})`,
            display: 'inline-block',
          }),
        },
        '插入工单回复'
      ),
      h(
        'span',
        {
          style: Object.assign(
            { display: 'flex', alignItems: 'center', gap: 6 },
            popStyle(done, '0 50%')
          ),
        },
        h(CheckCircle, { size: 18, color: TONE.blue.fg, p: seg(t, s + 2.1, s + 2.8) }),
        h('span', { style: sans(13.5, TONE.blue.fg) }, '已插入')
      )
    )
  )
}

const ENTRIES = [
  {
    icon: 'bubble',
    name: 'Zendesk 回票 agent',
    line: '回复客户之前，自动先查一遍知识库。',
    tone: TONE.ink,
  },
  {
    icon: 'stack',
    name: 'Dify 外部知识库',
    line: '挂进任意 Dify 应用，当知识库直接用。',
    tone: TONE.blue,
  },
]
function EntryPanel() {
  const t = useTime()
  const o = win(t, T.entry, T.secHard, 0.4, 0.4)
  if (o <= 0) return null
  return abs(
    { inset: 0, opacity: o },
    abs({ left: PANEL.x, top: 70 }, h('span', { style: sans(14, SUB) }, '在哪里用')),
    ENTRIES.map((e, k) => {
      const p = seg(t, T.entry + 0.2 + k * 0.5, T.entry + 0.7 + k * 0.5, E.easeOutCubic)
      return card(
        Object.assign(
          {
            key: e.name,
            left: PANEL.x,
            top: 102 + k * 128,
            width: PANEL.w,
            height: 112,
            padding: '0 22px',
            display: 'flex',
            alignItems: 'center',
            gap: 18,
          },
          rise(p, 12)
        ),
        h(
          'span',
          {
            style: {
              width: 52,
              height: 52,
              borderRadius: 12,
              background: e.tone.bg,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            },
          },
          h(Icon, {
            name: e.icon,
            size: 28,
            color: e.tone.fg,
            p: seg(t, T.entry + 0.4 + k * 0.5, T.entry + 1.3 + k * 0.5),
            t: t,
            sw: 1.6,
          })
        ),
        h(
          'div',
          null,
          h('div', { style: sans(19, INK, { fontWeight: 700 }) }, e.name),
          h('div', { style: sans(14, MUT, { marginTop: 4 }) }, e.line)
        )
      )
    })
  )
}

// =====================================================================
// ACT 4 — why it is hard (61.5–76): two columns, six problems → fixes
// =====================================================================
const COLS = [
  { title: '问题难回答', icon: 'question', tone: TONE.pink, x: 110, t: T.solveA },
  { title: '回答难被相信', icon: 'shield', tone: TONE.pink, x: 670, t: T.solveB },
]
const arrow = (k) => h('span', { key: k, style: sans(13, SUB) }, '→')
const ROWS = [
  {
    problem: '版本号、报错码混在一起',
    fix: '原样贴进去，直接搜',
    demo: [
      tag('#3412', TONE.gray, { mono: true, key: 'a' }),
      tag('3.8.0', TONE.blue, { mono: true, key: 'b' }),
      tag('panic: …', TONE.ink, { mono: true, key: 'c' }),
    ],
  },
  {
    problem: '症状描述很模糊',
    fix: '说「连不上」，也能找到报错',
    demo: [
      tag('连不上', TONE.blue, { key: 'a' }),
      arrow('x'),
      tag('ECONNREFUSED', TONE.green, { mono: true, key: 'b' }),
    ],
  },
  {
    problem: '答案藏在外部链接里',
    fix: '文档和 issue 一起找到',
    demo: [
      tag('docs/…', TONE.purple, { mono: true, key: 'a' }),
      tag('issue #232', TONE.purple, { mono: true, key: 'b' }),
    ],
  },
  {
    problem: '总结是怎么写出来的？',
    fix: '每条总结都说得清来历',
    demo: [
      tag('哪张票', TONE.gray, { key: 'a' }),
      arrow('x'),
      tag('哪次会话', TONE.gray, { key: 'b' }),
      arrow('y'),
      tag('哪条 prompt', TONE.gray, { key: 'c' }),
    ],
  },
  {
    problem: '原文到底是哪一句？',
    fix: '结论旁边就是原话',
    demo: [
      tag('结论', TONE.blue, { key: 'a' }),
      h('span', { key: 'x', style: sans(13, SUB) }, '↔'),
      tag('原话', TONE.ink, { key: 'b' }),
    ],
  },
  {
    problem: '个人信息会不会泄露？',
    fix: '入库前拦两遍',
    demo: [
      tag('[NAME_1]', TONE.pink, { mono: true, key: 'a' }),
      tag('[EMAIL_1]', TONE.pink, { mono: true, key: 'b' }),
    ],
  },
]
const ROW_T = 1.9
function Challenges() {
  const t = useTime()
  const o = win(t, T.hard, T.nums, 0.4, 0.5)
  if (o <= 0) return null
  return abs(
    { inset: 0, opacity: o },
    COLS.map((col, c) => {
      const hp = seg(t, T.hard + 0.1 + c * 0.35, T.hard + 0.7 + c * 0.35, E.easeOutCubic)
      // all three rows ticked → the column's icon resolves to a tick
      const done = seg(
        t,
        col.t + 0.3 + 2 * ROW_T + 0.5,
        col.t + 0.3 + 2 * ROW_T + 1.0,
        E.easeInOutCubic
      )
      return abs(
        Object.assign({ key: col.title, left: col.x, top: 96, width: 500 }, rise(hp, 10)),
        h(
          'div',
          { style: { display: 'flex', alignItems: 'center', gap: 14 } },
          h(
            'span',
            {
              style: {
                width: 44,
                height: 44,
                borderRadius: 10,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: `color-mix(in srgb, ${TONE.blue.bg} ${Math.round(done * 100)}%, ${
                  col.tone.bg
                })`,
              },
            },
            h(Icon, {
              name: col.icon,
              size: 26,
              color: done > 0.5 ? TONE.blue.fg : col.tone.fg,
              p: seg(t, T.hard + 0.2 + c * 0.35, T.hard + 1.1 + c * 0.35),
              done: done,
              sw: 1.7,
            })
          ),
          h(
            'span',
            { style: sans(30, INK, { fontWeight: 700, letterSpacing: '-0.01em' }) },
            col.title
          )
        ),
        h('div', { style: { height: 1, background: HAIR, marginTop: 20 } })
      )
    }),
    ROWS.map((r, i) => {
      const col = COLS[Math.floor(i / 3)],
        k = i % 3
      const pin = seg(t, T.hard + 0.6 + i * 0.18, T.hard + 1.1 + i * 0.18, E.easeOutCubic)
      const s = col.t + 0.3 + k * ROW_T
      const flip = seg(t, s, s + 0.45, E.easeInOutCubic)
      const demo = seg(t, s + 0.35, s + 0.8, E.easeOutCubic)
      const active = win(t, s - 0.15, s + ROW_T, 0.2, 0.3)
      return abs(
        Object.assign(
          {
            key: i,
            left: col.x - 14,
            top: 178 + k * 124,
            width: 528,
            height: 112,
            borderRadius: 8,
            background: `rgba(245,245,245,${active})`,
          },
          rise(pin, 8)
        ),
        abs({ left: 14, top: 18 }, h(Checkbox, { size: 22, p: seg(t, s, s + 0.6) })),
        abs(
          {
            left: 50,
            top: 14,
            opacity: 1 - seg(flip, 0, 0.5),
            transform: `translateY(${-6 * flip}px)`,
          },
          h('span', { style: sans(21, INK, { fontWeight: 500 }) }, r.problem)
        ),
        abs(
          {
            left: 50,
            top: 14,
            opacity: seg(flip, 0.5, 1),
            transform: `translateY(${6 * (1 - flip)}px)`,
          },
          h('span', { style: sans(21, INK, { fontWeight: 700 }) }, r.fix)
        ),
        abs(
          { left: 50, top: 48, opacity: flip },
          h('span', { style: sans(13.5, SUB, { textDecoration: 'line-through' }) }, r.problem)
        ),
        abs(
          Object.assign(
            { left: 50, top: 74, display: 'flex', gap: 6, alignItems: 'center' },
            rise(demo, 6)
          ),
          r.demo
        )
      )
    })
  )
}

// =====================================================================
// ACT 5 — numbers (76–88): two bar charts, two tiles
// =====================================================================
const SYSTEMS = ['本系统', '开源 GraphRAG 方案', '工单系统自带搜索']
const CHARTS = [
  {
    title: '命中率',
    note: '排在第一的就是那张票',
    icon: 'target',
    x: 100,
    vals: [0.711, 0.57, 0],
    max: 1,
    fmt: (v) => (v === 0 ? '0' : v.toFixed(3)),
    t: T.nums + 0.3,
  },
  {
    title: '查询中位延迟',
    note: '越短越好',
    icon: 'stopwatch',
    x: 680,
    vals: [0.8, 1.7, 26.4],
    max: 26.4,
    fmt: (v) => v + ' s',
    t: T.nums + 1.0,
  },
]
const TILES = [
  {
    big: '1/2',
    text: '查询延迟，对比开源 GraphRAG 方案',
    sub: '0.8 s vs 1.7 s（对方有缓存）',
    x: 390,
  },
]
function Numbers() {
  const t = useTime()
  const o = win(t, T.nums, T.values, 0.5, 0.5)
  if (o <= 0) return null
  const BAR = 250
  return abs(
    { inset: 0, opacity: o },
    CHARTS.map((c) => {
      const p = seg(t, c.t, c.t + 0.5, E.easeOutCubic)
      return abs(
        Object.assign({ key: c.title, left: c.x, top: 84, width: 500 }, rise(p, 10)),
        h(
          'div',
          { style: { display: 'flex', alignItems: 'center', gap: 12 } },
          h(Icon, {
            name: c.icon,
            size: 26,
            color: INK,
            accent: ACCENT,
            p: seg(t, c.t, c.t + 1.3),
            t: t,
            sw: 1.6,
          }),
          h('span', { style: sans(24, INK, { fontWeight: 700 }) }, c.title),
          h('span', { style: sans(14, SUB) }, c.note)
        ),
        h('div', { style: { height: 1, background: HAIR, margin: '14px 0 8px' } }),
        c.vals.map((v, k) => {
          const g = seg(t, c.t + 0.4 + k * 0.25, c.t + 1.4 + k * 0.25, E.easeOutCubic)
          const ours = k === 0
          return h(
            'div',
            { key: k, style: { display: 'flex', alignItems: 'center', height: 52 } },
            h(
              'span',
              { style: sans(15, ours ? INK : MUT, { width: 170, fontWeight: ours ? 600 : 400 }) },
              SYSTEMS[k]
            ),
            h(
              'div',
              {
                style: {
                  width: BAR,
                  height: 12,
                  borderRadius: 6,
                  background: '#f0f0f0',
                  position: 'relative',
                },
              },
              h('div', {
                style: {
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  height: 12,
                  borderRadius: 6,
                  width: Math.max(v > 0 ? 6 : 0, BAR * (v / c.max) * g),
                  background: ours ? ACCENT : FAINT,
                },
              })
            ),
            h(
              'span',
              {
                style: sans(20, ours ? ACCENT : MUT, {
                  fontWeight: 700,
                  marginLeft: 16,
                  opacity: g,
                  fontVariantNumeric: 'tabular-nums',
                }),
              },
              c.fmt(v)
            )
          )
        })
      )
    }),
    TILES.map((tile, k) => {
      const p = seg(t, T.nums2 + 0.3 + k * 0.45, T.nums2 + 0.9 + k * 0.45, E.easeOutCubic)
      return card(
        Object.assign(
          {
            key: tile.big,
            left: tile.x,
            top: 376,
            width: 500,
            height: 150,
            padding: '24px 28px',
            display: 'flex',
            alignItems: 'center',
            gap: 28,
            boxShadow: SH_SOFT,
          },
          rise(p, 12)
        ),
        h(
          'span',
          { style: serif(64, ACCENT, { fontWeight: 700, lineHeight: 1, width: 130 }) },
          tile.big
        ),
        h(
          'div',
          null,
          h('div', { style: sans(17, INK, { fontWeight: 600, lineHeight: 1.45 }) }, tile.text),
          h('div', { style: mono(12.5, SUB, { marginTop: 8 }) }, tile.sub)
        )
      )
    }),
    abs(
      {
        left: 0,
        right: 0,
        top: 556,
        textAlign: 'center',
        opacity: seg(t, T.nums + 1.6, T.nums + 2.2),
      },
      h(
        'span',
        { style: sans(13, SUB) },
        '2026-07 实测 · 同一组 149 个问题 · 本系统与 GraphRAG 同在 600 张工单上，自带搜索用其全量索引'
      )
    )
  )
}

// =====================================================================
// ACT 6 — three lines, serif (88–100), and "还有一件事" (100–101.6)
// =====================================================================
const VIGNETTE = { team: TeamVignette, repeat: RepeatVignette, ask: AskVignette }
const HEADS = [
  { t0: T.values + 0.3, t1: T.values + 3.8, big: '支持团队的经验，不再随人走。', art: 'team' },
  {
    t0: T.values + 4.3,
    t1: T.values + 7.8,
    big: '同一个缺陷第二次出现，就能被看见。',
    art: 'repeat',
  },
  { t0: T.values + 8.3, t1: T.values + 11.8, big: '下一张工单，先问它一句。', art: 'ask' },
  { t0: T.crystal + 0.1, t1: T.crystalDoc + 0.1, big: '还有一件事。' },
]
function Headline() {
  const t = useTime()
  const hd = HEADS.find((x) => t >= x.t0 && t <= x.t1)
  if (!hd) return null
  const o = win(t, hd.t0, hd.t1, 0.6, 0.5)
  return abs(
    {
      inset: 0,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 40,
      opacity: o,
      transform: `translateY(${(1 - seg(t, hd.t0, hd.t0 + 0.6, E.easeOutCubic)) * 12}px)`,
      pointerEvents: 'none',
    },
    hd.art
      ? h(
          'div',
          { style: { transform: 'scale(1.25)', margin: '18px 0 8px' } },
          h(VIGNETTE[hd.art], { p: seg(t, hd.t0 + 0.1, hd.t0 + 2.5), t: t - hd.t0 })
        )
      : null,
    h(
      'div',
      { style: serif(50, INK, { fontWeight: 700, letterSpacing: '0.01em', textAlign: 'center' }) },
      hd.big
    )
  )
}

// =====================================================================
// ACT 7 — one more thing (101.5–110): 14 tickets → one document
// =====================================================================
const CR = { s: T.crystalDoc, doc: { x: 380, y: 124, w: 520 } }
const CR_DOTS = (() => {
  const rng = mulberry32(7)
  const pts = []
  for (let i = 0; i < 14; i++) pts.push({ x: 150 + rng() * 320, y: 200 + rng() * 300 })
  // relax so the little ticket cards never overlap
  for (let it = 0; it < 60; it++)
    pts.forEach((a, i) =>
      pts.forEach((b, j) => {
        if (j <= i) return
        const dx = b.x - a.x,
          dy = b.y - a.y,
          d = Math.hypot(dx, dy) || 0.01
        if (d < 52) {
          const q = (52 - d) / 2 / d
          a.x -= dx * q
          a.y -= dy * q
          b.x += dx * q
          b.y += dy * q
        }
      })
    )
  return pts
})()
const CR_EDGES = (() => {
  const out = []
  for (let i = 0; i < CR_DOTS.length; i++)
    for (let j = i + 1; j < CR_DOTS.length; j++) {
      if (Math.hypot(CR_DOTS[i].x - CR_DOTS[j].x, CR_DOTS[i].y - CR_DOTS[j].y) < 120)
        out.push([i, j])
    }
  return out
})()
const CR_STEPS = [
  ['先定位是哪一层先断：比对各层生效值与时间戳', '#2286'],
  ['进运行中的 Pod 看实际值，不只看 Helm values', '#476'],
  ['查插件 Pod 的 SDK 版本，写死的走升级', '#1448'],
  ['Dify 侧都够了就往外查：ingress、WAF、外部 LB', '#2739'],
]
function Crystal() {
  const t = useTime()
  const s = CR.s
  const o = win(t, s, T.bento, 0.4, 0.5)
  if (o <= 0) return null
  const D = CR.doc
  const target = { x: D.x + D.w / 2, y: D.y + 150 }
  const gather = seg(t, s + 1.8, s + 2.7, E.easeInOutCubic)
  const docIn = seg(t, s + 2.3, s + 2.9, E.easeOutCubic)
  const els = []
  CR_EDGES.forEach(([i, j], k) => {
    const p = seg(t, s + 0.8 + k * 0.02, s + 1.4 + k * 0.02, E.easeInOutCubic) * (1 - gather)
    if (p <= 0) return
    const a = CR_DOTS[i],
      b = CR_DOTS[j]
    const len = Math.hypot(b.x - a.x, b.y - a.y)
    els.push(
      h('line', {
        key: 'e' + k,
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        stroke: EDGE,
        strokeWidth: 1.2,
        strokeDasharray: len,
        strokeDashoffset: len * (1 - p),
      })
    )
  })
  CR_DOTS.forEach((d, i) => {
    const ap = seg(t, s + 0.1 + i * 0.05, s + 0.5 + i * 0.05, E.easeOutBack)
    if (ap <= 0) return
    const g = seg(gather, i * 0.02, 0.72 + i * 0.02, E.easeInCubic)
    const bob = 2.5 * Math.sin(t * 1.6 + i * 0.9) * (1 - g)
    els.push(
      MiniTicket({
        key: 'd' + i,
        x: lerp(d.x, target.x, g),
        y: lerp(d.y, target.y, g) + bob,
        s: ap * (1 - 0.75 * g),
        o: 1 - g,
      })
    )
  })
  const lab = win(t, s + 0.3, s + 2.4, 0.4, 0.4)
  const stepIn = (i) => seg(t, s + 2.9 + i * 0.35, s + 3.3 + i * 0.35, E.easeOutCubic)
  const chipIn = (i) => seg(t, s + 3.1 + i * 0.35, s + 3.45 + i * 0.35, E.easeOutBack)
  const verified = seg(t, s + 4.6, s + 5.1, E.easeOutBack)
  return abs(
    { inset: 0, opacity: o },
    h('svg', { width: 1280, height: 720, style: { position: 'absolute', left: 0, top: 0 } }, els),
    abs(
      { left: 150, top: 160, opacity: lab },
      h('span', { style: sans(16, MUT) }, '同一类问题的 14 张工单')
    ),
    card(
      Object.assign({ left: D.x, top: D.y, width: D.w, padding: '24px 28px' }, rise(docIn, 16)),
      h(
        'div',
        { style: { display: 'flex', alignItems: 'center', gap: 10 } },
        h('span', { style: sans(13, SUB) }, '排查文档'),
        h('span', { style: { marginLeft: 'auto' } }, tag('实验中', TONE.blue, { size: 12 }))
      ),
      h(
        'div',
        { style: sans(22, INK, { fontWeight: 700, marginTop: 8 }) },
        '插件执行超时：先确定哪一层先断'
      ),
      h('div', { style: sans(13, SUB, { marginTop: 6 }) }, '由 14 张已关闭工单合成'),
      divider('d'),
      CR_STEPS.map(([text, id], i) =>
        h(
          'div',
          {
            key: id,
            style: Object.assign(
              { display: 'flex', alignItems: 'center', gap: 10, minHeight: 34 },
              rise(stepIn(i), 8)
            ),
          },
          h('span', { style: sans(14, SUB, { width: 18 }) }, i + 1 + '.'),
          h('span', { style: sans(14.5, INK, { flex: 1, lineHeight: 1.45 }) }, text),
          h(
            'span',
            { style: popStyle(chipIn(i), '100% 50%') },
            tag(id, TONE.blue, { mono: true, size: 12 })
          )
        )
      ),
      h(
        'div',
        {
          style: {
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginTop: 12,
            opacity: clamp01(verified * 1.4),
          },
        },
        h(CheckCircle, { size: 18, color: TONE.blue.fg, p: seg(t, s + 4.6, s + 5.3) }),
        h('span', { style: sans(13, MUT) }, '28 条出处，逐条核对')
      )
    )
  )
}

// =====================================================================
// ACT 8 — the bento (110–120)
// =====================================================================
const BG = { x: 80, y: 64, w: 1120, h: 560, gap: 14, cols: 5, rows: 3 }
const bColW = (BG.w - (BG.cols - 1) * BG.gap) / BG.cols
const bRowH = (BG.h - (BG.rows - 1) * BG.gap) / BG.rows
const bRect = (c, r, cs, rs) => ({
  left: BG.x + c * (bColW + BG.gap),
  top: BG.y + r * (bRowH + BG.gap),
  width: cs * bColW + (cs - 1) * BG.gap,
  height: rs * bRowH + (rs - 1) * BG.gap,
})
const HERO = bRect(0, 0, 2, 2)
const HERO_CAM = {
  cx: 500,
  cy: 280,
  s: 0.38,
  ox: HERO.left + HERO.width / 2,
  oy: HERO.top + HERO.height - 108,
}
const BENTO = [
  { c: 0, r: 0, cs: 2, rs: 2, kind: 'hero' },
  { c: 2, r: 0, cs: 1, rs: 1, kind: 'num', big: '3,400+', text: '张已关闭工单，连成一张图' },
  { c: 3, r: 0, cs: 1, rs: 1, kind: 'num', big: '0.711', text: '命中率：排第一的就是那张票' },
  { c: 2, r: 1, cs: 1, rs: 1, kind: 'num', big: '0.8 s', text: '查询中位延迟，检索不调用 LLM' },
  { c: 3, r: 1, cs: 1, rs: 1, kind: 'num', big: '1/2', text: '查询延迟，对比开源 GraphRAG 方案' },
  { c: 4, r: 0, cs: 1, rs: 2, kind: 'next' },
  {
    c: 0,
    r: 2,
    cs: 1,
    rs: 1,
    kind: 'feat',
    title: '每句结论带原话',
    icon: 'quote',
    tone: TONE.ink,
    body: [
      tag('结论', TONE.blue, { key: 'a', size: 12 }),
      h('span', { key: 'x', style: sans(12, SUB) }, '↔'),
      tag('原话', TONE.ink, { key: 'b', size: 12 }),
    ],
  },
  {
    c: 1,
    r: 2,
    cs: 1,
    rs: 1,
    kind: 'feat',
    title: '脱敏两道关',
    icon: 'shield',
    tone: TONE.pink,
    body: [
      tag('[NAME_1]', TONE.pink, { key: 'a', mono: true, size: 11.5 }),
      tag('[EMAIL_1]', TONE.pink, { key: 'b', mono: true, size: 11.5 }),
    ],
  },
  {
    c: 2,
    r: 2,
    cs: 1,
    rs: 1,
    kind: 'feat',
    title: '每天自动同步',
    icon: 'moon',
    tone: TONE.blue,
    body: [h('span', { key: 'a', style: sans(13, MUT) }, '解决的工单，次日入图')],
  },
  {
    c: 3,
    r: 2,
    cs: 2,
    rs: 1,
    kind: 'feat',
    title: '两个入口',
    body: ENTRIES.map((e) =>
      h(
        'span',
        { key: e.name, style: { display: 'flex', alignItems: 'center', gap: 8, marginRight: 14 } },
        h(
          'span',
          {
            style: {
              width: 26,
              height: 26,
              borderRadius: 7,
              background: e.tone.bg,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            },
          },
          h(Icon, { name: e.icon, size: 16, color: e.tone.fg, sw: 1.8 })
        ),
        h('span', { style: sans(14, INK) }, e.name)
      )
    ),
  },
]
// bento hero: the graph at rest, with a few pulses still running along it
function GraphStatic({ cam, t }) {
  const els = []
  const pos = G.tickets.map((tk) => toScreen(cam, tk.x, tk.y))
  const fpos = G.feats.map((f) => toScreen(cam, f.x, f.y))
  G.edges.forEach((e, k) => {
    const a = pos[e.t],
      b = fpos[e.f]
    els.push(
      h('line', {
        key: 'e' + k,
        x1: a[0],
        y1: a[1],
        x2: b[0],
        y2: b[1],
        stroke: EDGE,
        strokeWidth: 0.8,
      })
    )
  })
  const live = seg(t, T.bento + 1.2, T.bento + 2.2)
  G.edges.forEach((e, k) => {
    if (k % 9 !== 0 || live <= 0) return
    const q = cycle(t, 2.6, k * 0.173)
    const a = pos[e.t],
      b = fpos[e.f]
    els.push(
      h('circle', {
        key: 'p' + k,
        cx: lerp(a[0], b[0], q),
        cy: lerp(a[1], b[1], q),
        r: 1.8,
        fill: KIND_TONE[G.feats[e.f].kind].fg,
        opacity: live * bell(q),
      })
    )
  })
  fpos.forEach((p, j) => {
    els.push(
      h('circle', {
        key: 'f' + j,
        cx: p[0],
        cy: p[1],
        r: 3.4,
        fill: KIND_TONE[G.feats[j].kind].fg,
        stroke: '#fff',
        strokeWidth: 1.2,
      })
    )
  })
  pos.forEach((p, i) => {
    els.push(
      h('circle', {
        key: 't' + i,
        cx: p[0],
        cy: p[1],
        r: i === 0 ? 3.4 : 2.2,
        fill: i === 0 ? ACCENT : INK,
        fillOpacity: i === 0 ? 1 : 0.7,
      })
    )
  })
  const ring = cycle(t, 2)
  els.push(
    h('circle', {
      key: 'ring',
      cx: pos[0][0],
      cy: pos[0][1],
      r: 3.4 + 10 * ring,
      fill: 'none',
      stroke: ACCENT,
      strokeWidth: 1,
      opacity: live * (1 - ring),
    })
  )
  return h(
    'svg',
    {
      width: 1280,
      height: 720,
      style: { position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none' },
    },
    els
  )
}
function Bento() {
  const t = useTime()
  const o = win(t, T.bento, END + 1, 0.4, 0.3)
  if (o <= 0) return null
  return abs(
    { inset: 0, opacity: o },
    BENTO.map((b, k) => {
      const p = seg(t, T.bento + 0.2 + k * 0.14, T.bento + 0.75 + k * 0.14, E.easeOutCubic)
      const base = Object.assign(
        { boxShadow: SH_SOFT, borderRadius: 12 },
        bRect(b.c, b.r, b.cs, b.rs),
        rise(p, 14)
      )
      if (b.kind === 'hero')
        return card(
          Object.assign({ key: 'hero', padding: '30px 34px', overflow: 'hidden' }, base),
          h(GraphStatic, {
            t: t,
            cam: Object.assign({}, HERO_CAM, {
              ox: HERO_CAM.ox - HERO.left,
              oy: HERO_CAM.oy - HERO.top,
            }),
          }),
          h(
            'div',
            {
              style: sans(15, SUB, {
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }),
            },
            h(Icon, {
              name: 'graph',
              size: 18,
              p: seg(t, T.bento + 0.3, T.bento + 1.2),
              t: t,
              sw: 1.8,
            }),
            '历史工单知识库'
          ),
          h(
            'div',
            {
              style: serif(34, INK, {
                position: 'relative',
                fontWeight: 700,
                lineHeight: 1.35,
                marginTop: 14,
              }),
            },
            '每一张关闭的工单，',
            h('br'),
            '都是下一张的答案。'
          )
        )
      if (b.kind === 'num')
        return card(
          Object.assign(
            {
              key: b.big,
              padding: '22px 22px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
            },
            base
          ),
          h(
            'div',
            {
              style: sans(40, INK, {
                fontWeight: 700,
                letterSpacing: '-0.02em',
                lineHeight: 1,
                whiteSpace: 'nowrap',
              }),
            },
            b.big
          ),
          h('div', { style: sans(13, MUT, { lineHeight: 1.5 }) }, b.text)
        )
      if (b.kind === 'next')
        return card(
          Object.assign(
            {
              key: 'next',
              padding: '22px 22px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              background: TONE.blue.bg,
            },
            base
          ),
          h(
            'div',
            null,
            tag('实验中', { fg: TONE.blue.fg, bg: '#fff' }, { size: 12 }),
            h(
              'div',
              { style: sans(18, INK, { fontWeight: 700, lineHeight: 1.4, marginTop: 10 }) },
              '同类工单，自动合成一篇文档'
            )
          ),
          h(DocGather, { t: t - T.bento }),
          h('div', { style: sans(13, MUT) }, '14 张票 → 1 篇排查文档')
        )
      return card(
        Object.assign(
          {
            key: b.title,
            padding: '22px 22px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          },
          base
        ),
        h(
          'div',
          { style: { display: 'flex', alignItems: 'center', gap: 8 } },
          b.icon
            ? h(Icon, {
                name: b.icon,
                size: 20,
                color: b.tone.fg,
                accent: b.tone.fg,
                p: seg(t, T.bento + 0.6 + k * 0.14, T.bento + 1.6 + k * 0.14),
                t: t,
                done: 1,
                sw: 1.7,
              })
            : null,
          h('span', { style: sans(17, INK, { fontWeight: 700 }) }, b.title)
        ),
        h(
          'div',
          { style: { display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' } },
          b.body
        )
      )
    }),
    abs(
      {
        left: 0,
        right: 0,
        top: 646,
        textAlign: 'center',
        opacity: seg(t, T.bento + 2.0, T.bento + 2.6),
      },
      h(
        'span',
        { style: sans(12.5, SUB) },
        '示例为真实已关闭工单，客户信息已去除 · 对比：2026-07 实测，同一组 149 个问题，本系统与 GraphRAG 同在 600 张工单上'
      )
    )
  )
}

// =====================================================================
// captions
// =====================================================================
const CAPS = [
  [T.ticket + 0.3, T.buried - 0.3, '一张新工单：升级到 3.9.8 后，代码节点崩了。'],
  [T.buried + 0.4, T.connect - 0.2, '答案在半年前一张已关闭的工单里。没人找得到。'],
  [T.connect + 0.4, T.promise - 0.2, '把已关闭的工单，按关键词、版本、链接连成一张图。'],
  [T.promise + 0.4, T.secForm - 0.2, '问一句，返回排好序的旧工单，每句结论带原话出处。'],
  [T.form + 0.4, T.extract - 0.2, '先脱敏：人名、邮箱换成占位符，报错原文留着。'],
  [T.extract + 0.3, T.nodes - 0.2, '再提炼：一句结论、几个关键词、版本号；结论指回原话。'],
  [T.nodes + 0.3, T.grow - 0.2, '它们落成图里的节点和边。'],
  [T.grow + 0.2, T.secSearch - 0.3, '共用关键词的工单自动相连；每天夜里，新关的工单加入。'],
  [T.query + 0.4, T.channels - 0.2, '还是开头那张工单，原话直接贴进去。'],
  [T.channels + 0.3, T.rank - 0.3, '四路同时找：语义、原文、关键词、图关系。'],
  [T.rank + 0.3, T.answer - 0.2, '几路结果合起来排序：#2948 排第一。'],
  [T.answer + 0.3, T.entry - 0.2, '结论旁边附原话出处，直接回进工单。'],
  [T.entry + 0.3, T.secHard - 0.3, 'Zendesk 回票时自动先查；也能挂进 Dify 应用。'],
  [T.solveA + 0.3, T.solveB - 0.3, '难回答：原样贴、说症状、找链接，都能搜到。'],
  [T.solveB + 0.3, T.nums - 0.3, '难相信：每条总结有来历，每句结论有原话，隐私拦两遍。'],
  [T.nums + 0.4, T.nums2 - 0.1, '命中率 0.711，开源 GraphRAG 方案 0.570，工单系统自带搜索是 0。'],
  [T.nums2 + 0.3, T.values - 0.3, '查询延迟减半，检索全程不调用大模型。'],
  [T.crystalDoc + 0.4, T.bento - 0.3, '同类问题的 14 张工单，自动合成一篇排查文档，每句都带出处。'],
]
function Caption() {
  const t = useTime()
  const cap = CAPS.find(([a, b]) => t >= a && t <= b)
  if (!cap) return null
  const o = win(t, cap[0], cap[1], 0.4, 0.4)
  return abs(
    {
      left: 0,
      right: 0,
      top: 652,
      textAlign: 'center',
      opacity: o,
      transform: `translateY(${(1 - seg(t, cap[0], cap[0] + 0.4, E.easeOutCubic)) * 8}px)`,
    },
    h('span', { style: sans(20, MUT, { letterSpacing: '0.01em' }) }, cap[2])
  )
}

// ?t=SECONDS opens the film paused at that second
const readSeek = () => {
  try {
    const v = parseFloat(new URLSearchParams(window.location.search).get('t'))
    return isFinite(v) ? Math.max(0, Math.min(END, v)) : null
  } catch (e) {
    return null
  }
}

// startAt: open at this second instead of the persisted playhead.
// paused: hold playback (e.g. while scrolled away); resumes if it was playing.
export default function EventFilm({ showCaptions = true, startAt, paused = false }) {
  const [seek] = useState(readSeek)
  const [reduced] = useState(prefersReducedMotion)
  return h(
    Stage,
    {
      width: 1280,
      height: 720,
      duration: END,
      background: '#ffffff',
      persistKey: 'hb-event-film',
      autoplay: seek == null && !reduced,
      loop: true,
      startAt: seek == null ? startAt : seek,
      paused: paused,
    },
    h(Sprite, { start: 0, end: T.buried + 0.1 }, h(SupportIntro), h(NewTicket)),
    h(Sprite, { start: T.buried, end: T.secForm + 0.1 }, h(BuriedTag), h(CountUp), h(PromiseScene)),
    h(Sprite, { start: T.form, end: T.secSearch + 0.1 }, h(FormPage), h(ExtractPanel)),
    h(GraphLayer),
    h(Sprite, { start: T.nodes, end: T.nodes + 1.4 }, h(FlyingChips)),
    h(
      Sprite,
      { start: T.query, end: T.secHard + 0.1 },
      h(QueryBox),
      h(ChannelPanel),
      h(RankPanel),
      h(AnswerPanel),
      h(EntryPanel)
    ),
    h(Sprite, { start: T.hard, end: T.nums + 0.1 }, h(Challenges)),
    h(Sprite, { start: T.nums, end: T.values + 0.1 }, h(Numbers)),
    h(Sprite, { start: T.crystalDoc, end: T.bento + 0.1 }, h(Crystal)),
    h(Sprite, { start: T.bento, end: END }, h(Bento)),
    h(Headline),
    showCaptions ? h(Caption) : null,
    h(SectionTitle)
  )
}
