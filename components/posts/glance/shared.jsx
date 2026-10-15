// Shared helpers for the glance post figures (components/posts/glance/figures).
// Ported from the inline script of external-blogs/glance-version/blog.html.
import { useEffect, useRef, useState } from 'react'

export const clamp01 = (v) => Math.max(0, Math.min(1, v))
export const lerp = (a, b, p) => a + (b - a) * p
export const easeInOut = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2)

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  !!window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches

/**
 * Tone → [ink, wash], resolved through the css/glance.css variables (`.glance` scope).
 * DESIGN.md set: gray / ink neutrals, the one theme blue, Warning Pink. green and purple
 * exist only for the feature kinds the Sea caption names (关键词绿 / 外部链接紫).
 */
export const TONE = {
  gray: ['var(--gray)', 'var(--gray-bg)'],
  ink: ['var(--ink)', 'var(--gray-bg)'],
  blue: ['var(--blue)', 'var(--blue-bg)'],
  pink: ['var(--pink)', 'var(--pink-bg)'],
  green: ['var(--green)', 'var(--green-bg)'],
  purple: ['var(--purple)', 'var(--purple-bg)'],
}

/** `<span class="tag {tone} [mono]">` from blog.html. */
export function Tag({ tone = 'gray', mono = false, children }) {
  return <span className={`tag ${tone}${mono ? ' mono' : ''}`}>{children}</span>
}

/**
 * Fires `fn` (or flips `revealed`) once when the element scrolls into view.
 * Replaces blog.html's `onReveal(node, fn, threshold)`.
 */
export function useReveal(threshold = 0.3) {
  const ref = useRef(null)
  const [revealed, setRevealed] = useState(false)
  useEffect(() => {
    const node = ref.current
    if (!node || revealed) return
    if (!('IntersectionObserver' in window)) {
      setRevealed(true)
      return
    }
    const io = new IntersectionObserver(
      (es) => {
        if (es.some((e) => e.isIntersecting)) {
          io.disconnect()
          setRevealed(true)
        }
      },
      { threshold }
    )
    io.observe(node)
    return () => io.disconnect()
  }, [threshold, revealed])
  return [ref, revealed]
}

/** Deterministic PRNG so the illustrative graph is identical on every render. */
export function mulberry32(a) {
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let q = Math.imul(a ^ (a >>> 15), 1 | a)
    q = (q + Math.imul(q ^ (q >>> 7), 61 | q)) ^ q
    return ((q ^ (q >>> 14)) >>> 0) / 4294967296
  }
}

export const shuffle = (arr, rng) => {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const x = arr[i]
    arr[i] = arr[j]
    arr[j] = x
  }
  return arr
}

// =====================================================================
// the graph (same construction as the film: 14 features, 78 tickets)
// Used by <Glance.Sea /> and <Glance.Graph />.
// =====================================================================
export const FEATS = [
  { key: 'sandbox', kind: 'kw', c: 0, x: 330, y: 250 },
  { key: 'seccomp', kind: 'kw', c: 0, x: 440, y: 320 },
  { key: '3.9.x', kind: 'ver', c: 0, x: 310, y: 370 },
  { key: 'dify-sandbox #232', kind: 'link', c: 0, x: 455, y: 200 },
  { key: 'plugin', kind: 'kw', c: 1, x: 680, y: 110 },
  { key: 'timeout', kind: 'kw', c: 1, x: 805, y: 160 },
  { key: 'helm', kind: 'kw', c: 1, x: 665, y: 225 },
  { key: 'redis', kind: 'kw', c: 2, x: 760, y: 390 },
  { key: 'celery', kind: 'kw', c: 2, x: 880, y: 445 },
  { key: '3.8.x', kind: 'ver', c: 2, x: 700, y: 475 },
  { key: 'sso', kind: 'kw', c: 3, x: 135, y: 125 },
  { key: 'oauth', kind: 'kw', c: 3, x: 250, y: 90 },
  { key: 'rag', kind: 'kw', c: 4, x: 140, y: 415 },
  { key: 'embedding', kind: 'kw', c: 4, x: 235, y: 490 },
]
export const KIND = {
  kw: { tone: 'green', name: ['Keyword', '关键词'] },
  ver: { tone: 'blue', name: ['Version', '版本'] },
  link: { tone: 'purple', name: ['External link', '外部链接'] },
}
export const fIdx = (key) => FEATS.findIndex((f) => f.key === key)

export const G = (() => {
  const W = 1000,
    H = 560
  const rng = mulberry32(924)
  const feats = FEATS.map((f, i) => Object.assign({ i: i }, f))
  const own = (c) => feats.filter((f) => f.c === c).map((f) => f.i)
  const NB = [
    [1, 2, 4],
    [0, 2],
    [0, 1],
    [0, 4],
    [0, 3],
  ]
  const SIZES = [18, 16, 15, 13, 16]
  const tickets = [
    {
      id: '2948',
      c: 0,
      fs: [fIdx('sandbox'), fIdx('seccomp'), fIdx('3.9.x')],
      title: [
        'Code execution fails after upgrading dify to 3.9.5',
        'dify 升级到 3.9.5 后代码执行报错',
      ],
    },
    {
      id: '3256',
      c: 0,
      fs: [fIdx('sandbox'), fIdx('3.9.x'), fIdx('dify-sandbox #232')],
      title: ['Sandbox unable to run in v3.9.5, …', 'Sandbox unable to run in v3.9.5, …'],
    },
  ]
  SIZES.forEach((n, c) => {
    for (let k = c === 0 ? 2 : 0; k < n; k++) {
      const pool = shuffle(own(c), rng)
      const fs = pool.slice(0, rng() < 0.25 ? 1 : 2)
      if (rng() < 0.3) {
        const nb = own(NB[c][Math.floor(rng() * NB[c].length)])
        fs.push(nb[Math.floor(rng() * nb.length)])
      }
      tickets.push({ id: null, c: c, fs: fs })
    }
  })
  const used = new Set(['2948', '3256'])
  tickets.forEach((tk) => {
    while (!tk.id) {
      const id = String(180 + Math.floor(rng() * 3200))
      if (!used.has(id)) {
        tk.id = id
        used.add(id)
      }
    }
  })
  tickets.forEach((tk, i) => {
    tk.mx = tk.fs.reduce((s, f) => s + feats[f].x, 0) / tk.fs.length
    tk.my = tk.fs.reduce((s, f) => s + feats[f].y, 0) / tk.fs.length
    const a = rng() * Math.PI * 2,
      r = i === 0 ? 0 : (tk.fs.length === 1 ? 34 : 16) + rng() * 44
    tk.x = tk.mx + Math.cos(a) * r
    tk.y = tk.my + Math.sin(a) * r
  })
  for (let it = 0; it < 160; it++) {
    for (let i = 0; i < tickets.length; i++) {
      const a = tickets[i]
      for (let j = i + 1; j < tickets.length; j++) {
        const b = tickets[j]
        const dx = b.x - a.x,
          dy = b.y - a.y,
          d = Math.hypot(dx, dy) || 0.01
        if (d < 25) {
          const p = (25 - d) / 2 / d
          a.x -= dx * p
          a.y -= dy * p
          b.x += dx * p
          b.y += dy * p
        }
      }
      feats.forEach((f) => {
        const dx = a.x - f.x,
          dy = a.y - f.y,
          d = Math.hypot(dx, dy) || 0.01
        if (d < 30) {
          const p = (30 - d) / d
          a.x += dx * p
          a.y += dy * p
        }
      })
      a.x += (a.mx - a.x) * 0.015
      a.y += (a.my - a.y) * 0.015
      a.x = Math.max(24, Math.min(W - 24, a.x))
      a.y = Math.max(24, Math.min(H - 24, a.y))
    }
  }
  const order = shuffle(
    tickets.map((_, i) => i),
    rng
  )
  const k0 = order.indexOf(0),
    CENTRE = 2 * 13 + 5
  order[k0] = order[CENTRE]
  order[CENTRE] = 0
  order.forEach((ti, k) => {
    const col = k % 13,
      row = Math.floor(k / 13)
    tickets[ti].x0 = (col + 0.5) * (W / 13) + (rng() - 0.5) * 34
    tickets[ti].y0 = (row + 0.5) * (H / 6) + (rng() - 0.5) * 38
  })
  tickets.forEach((tk) => {
    tk.d1 = rng()
    tk.d2 = rng()
  })
  const edges = []
  tickets.forEach((tk, i) => tk.fs.forEach((f) => edges.push({ t: i, f: f })))
  const byFeat = feats.map(() => [])
  tickets.forEach((tk, i) => tk.fs.forEach((f) => byFeat[f].push(i)))
  feats.forEach((f, j) => {
    f.deg = byFeat[j].length
    f.rarity = 1 / Math.log2(1 + f.deg)
  })
  return { W: W, H: H, tickets: tickets, feats: feats, edges: edges, byFeat: byFeat }
})()

export const featR = (f) => 5 + 1.25 * Math.sqrt(f.deg)
