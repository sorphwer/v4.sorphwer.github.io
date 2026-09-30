/**
 * Seeded generative artwork for post banners and related-post cards, emitted as
 * an SVG string so it is part of the statically generated HTML (see PostArt).
 * The composition is derived from the post slug alone, so a post always gets
 * the same picture.
 *
 * Output must be byte-identical on the server and in the browser (hydration),
 * so everything here is integer/IEEE arithmetic: a seeded PRNG, hash-based
 * value noise, a rounded direction table instead of per-step trig, and every
 * coordinate rounded to an integer.
 *
 * Colours are not baked in: shapes carry role classes (`f*` fill, `s*` stroke)
 * that css/tailwind.css maps to DESIGN.md's palette per theme. Light: white
 * ground, black ink, RS Blue Light; dark: black ground, white ink, RS Blue
 * Dark; 50% Grey and Warning Pink in both. Roles: g ground, i ink, m grey,
 * b blue, p pink.
 */

/** Artwork height in SVG units; boxes of any size show a centred `slice` of it. */
export const ART_HEIGHT = 256

/** FNV-1a: stable 32-bit seed from a slug. */
function hashSeed(text) {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** mulberry32: small seeded PRNG in [0, 1). */
function makeRandom(seed) {
  let a = seed
  const random = () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  random.range = (lo, hi) => lo + random() * (hi - lo)
  random.int = (n) => Math.floor(random() * n)
  return random
}

/** Three-octave value noise in [0, 1), p5 `noise()`-like. */
function makeNoise(seed) {
  const lattice = (x, y) => {
    let h = Math.imul(x, 0x27d4eb2d) ^ Math.imul(y, 0x165667b1) ^ seed
    h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d)
    h ^= h >>> 12
    return (h >>> 0) / 4294967296
  }
  const smooth = (t) => t * t * (3 - 2 * t)
  const layer = (x, y) => {
    const xi = Math.floor(x)
    const yi = Math.floor(y)
    const u = smooth(x - xi)
    const v = smooth(y - yi)
    const a = lattice(xi, yi)
    const b = lattice(xi + 1, yi)
    const c = lattice(xi, yi + 1)
    const d = lattice(xi + 1, yi + 1)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
  }
  return (x, y) => (layer(x, y) * 4 + layer(x * 2, y * 2) * 2 + layer(x * 4, y * 4)) / 7
}

// 64 unit directions, rounded so platform trig differences cannot leak into output.
const DIRECTIONS = Array.from({ length: 64 }, (_, i) => [
  Math.round(Math.cos((i * Math.PI) / 32) * 1e6) / 1e6,
  Math.round(Math.sin((i * Math.PI) / 32) * 1e6) / 1e6,
])

const n = Math.round
const pt = (x, y) => `${n(x)} ${n(y)}`
const polygon = (points) => `M${points.map(([x, y]) => pt(x, y)).join('L')}Z`
const arc = (x0, y0, x1, y1, radius) =>
  `M${pt(x0, y0)}A${n(radius)} ${n(radius)} 0 0 1 ${pt(x1, y1)}`
const circle = (cx, cy, radius) =>
  `M${pt(cx - radius, cy)}a${n(radius)} ${n(radius)} 0 1 0 ${n(2 * radius)} 0a${n(radius)} ${n(
    radius
  )} 0 1 0 ${-n(2 * radius)} 0Z`

/**
 * Shapes batched into one <path> per style, so a banner is a few dozen
 * elements instead of hundreds. Batching reorders drawing, so patterns whose
 * shapes overlap across styles use separate layers.
 */
function makeLayer() {
  const groups = new Map()
  const add = (key, attrs, d) => {
    let group = groups.get(key)
    if (!group) groups.set(key, (group = { attrs, d: [] }))
    group.d.push(d)
  }
  return {
    fill: (role, d) => add(`f${role}`, `class="f${role}"`, d),
    stroke: (role, attrs, d) => add(`s${role}${attrs}`, `class="s${role}" ${attrs}`, d),
    toString: () =>
      [...groups.values()].map((g) => `<path ${g.attrs} d="${g.d.join('')}"/>`).join(''),
  }
}

/** Base (grey / ink) : blue : pink ≈ 6 : 3 : 1 over the drawn shapes. */
function pickInk(random) {
  const r = random()
  if (r < 0.6) return r < 0.4 ? 'm' : 'i'
  if (r < 0.9) return 'b'
  return 'p'
}

/**
 * Interlocking isometric L blocks, op-art style: chevron lids, grey left and
 * ink right L faces tile the plane with no gaps. Grid point (i, j) is
 * i·(down-right) + j·(down-left) unit edges; each 3 × 3 lattice cell holds one
 * lid and one L of each side. Lids are ground (a few blue / pink), so ground
 * lids are left to the background rect.
 */
function blocks({ width, height, random }) {
  const layer = makeLayer()
  const u = height / random.range(9, 14)
  const rx = (u * Math.sqrt(3)) / 2
  const shape = (i0, j0, points) =>
    polygon(points.map(([i, j]) => [(i0 + i - j0 - j) * rx, ((i0 + i + j0 + j) * u) / 2]))
  // Lattice points (3m, 3n): x = 3(m − n)·rx, y = 3(m + n)·u/2; m − n and m + n
  // share parity. Pieces reach 3·rx either side and 5.5·u below their point.
  for (let t = -3; t * 1.5 * u <= height; t++) {
    for (let s = -2; s * 3 * rx <= width + 3 * rx; s++) {
      if ((s + t) % 2) continue
      const i0 = ((t + s) / 2) * 3
      const j0 = ((t - s) / 2) * 3
      const pick = random()
      const lid = pick < 0.16 ? 'b' : pick < 0.2 ? 'p' : 'g'
      if (lid !== 'g') {
        layer.fill(
          lid,
          shape(i0, j0, [
            [0, 0],
            [2, 0],
            [2, 1],
            [1, 1],
            [1, 2],
            [0, 2],
          ])
        )
      }
      layer.fill(
        'm',
        shape(i0, j0, [
          [1, 4],
          [2, 4],
          [3, 5],
          [4, 5],
          [5, 6],
          [3, 6],
        ])
      )
      layer.fill(
        'i',
        shape(i0, j0, [
          [4, 2],
          [4, 1],
          [6, 3],
          [6, 5],
          [5, 4],
          [5, 3],
        ])
      )
    }
  }
  return String(layer)
}

/** Truchet quarter-arcs: continuous grey strands, some cells re-inked. */
function truchet({ width, height, random }) {
  const layer = makeLayer()
  const size = height / Math.floor(random.range(3, 6))
  const h = size / 2
  const attrs = `stroke-width="${n(size * 0.2)}"`
  for (let y = 0; y < height; y += size) {
    for (let x = 0; x < width; x += size) {
      const role = random() < 0.55 ? 'm' : pickInk(random)
      const d =
        random() < 0.5
          ? arc(x + h, y, x, y + h, h) + arc(x + h, y + size, x + size, y + h, h)
          : arc(x + size, y + h, x + h, y, h) + arc(x, y + h, x + h, y + size, h)
      layer.stroke(role, attrs, d)
    }
  }
  return String(layer)
}

/** Bauhaus tiles: each cell one primitive (quarter / half disc, circle, triangle, bars). */
function bauhaus({ width, height, random }) {
  const backs = makeLayer()
  const fronts = makeLayer()
  const size = height / Math.floor(random.range(2, 4))
  const h = size / 2
  for (let y = 0; y < height; y += size) {
    for (let x = 0; x < width; x += size) {
      const back = random() < 0.55 ? 'g' : pickInk(random)
      let front = pickInk(random)
      if (front === back) front = back === 'g' ? 'm' : 'g'
      backs.fill(back, `M${pt(x, y)}h${n(size)}v${n(size)}h${-n(size)}Z`)
      const turn = random.int(4)
      const kind = random.int(6)
      // Cell-local point, rotated by quarter turns (exact), to canvas space.
      const at = (u, v) => {
        for (let i = 0; i < turn; i++) [u, v] = [-v, u]
        return [x + h + u, y + h + v]
      }
      const p = (u, v) => pt(...at(u, v))
      if (kind === 0) {
        fronts.fill(front, `M${p(-h, -h)}L${p(h, -h)}A${n(size)} ${n(size)} 0 0 1 ${p(-h, h)}Z`)
      } else if (kind === 1) {
        fronts.fill(front, `M${p(-h, h)}A${n(h)} ${n(h)} 0 0 1 ${p(h, h)}Z`)
      } else if (kind === 2) {
        fronts.fill(front, circle(x + h, y + h, size * 0.31))
      } else if (kind === 3) {
        fronts.fill(front, polygon([at(-h, -h), at(h, -h), at(-h, h)]))
      } else if (kind === 4) {
        for (let i = 0; i < 3; i++) {
          const top = -h + (i * size) / 3
          const bottom = top + size / 7
          fronts.fill(front, polygon([at(-h, top), at(h, top), at(h, bottom), at(-h, bottom)]))
        }
      }
    }
  }
  return String(backs) + String(fronts)
}

/** Noise flow field: thin grey strands with blue and pink currents. */
function flow({ width, height, random, noise }) {
  const layer = makeLayer()
  const k = random.range(0.002, 0.005)
  const twist = random.range(1.5, 3)
  const count = Math.floor((width * height) / 2400)
  const step = 10
  for (let i = 0; i < count; i++) {
    let x = random.range(-40, width + 40)
    let y = random.range(-40, height + 40)
    const start = pt(x, y)
    const role = pickInk(random)
    const opacity = random() < 0.5 ? 0.55 : 0.9
    const strokeWidth = [1, 1.5, 2.5][random.int(3)]
    const steps = Math.floor(random.range(10, 35))
    const moves = []
    for (let j = 0; j < steps; j++) {
      const dir = DIRECTIONS[((Math.floor(noise(x * k, y * k) * twist * 64) % 64) + 64) % 64]
      const nx = x + dir[0] * step
      const ny = y + dir[1] * step
      moves.push(`${n(nx) - n(x)} ${n(ny) - n(y)}`)
      x = nx
      y = ny
    }
    layer.stroke(
      role,
      `stroke-width="${strokeWidth}" stroke-opacity="${opacity}" stroke-linejoin="round"`,
      `M${start}l${moves.join(' ')}`
    )
  }
  return String(layer)
}

/** Halftone: a dot grid whose sizes follow a noise wave; coloured bands ride the crests. */
function halftone({ width, height, random, noise }) {
  const layer = makeLayer()
  const step = height / random.range(8, 12)
  const k = random.range(0.004, 0.01)
  for (let y = step / 2; y < height; y += step) {
    for (let x = step / 2; x < width; x += step) {
      const v = noise(x * k, y * k)
      const d = Math.round((step * 0.95 * Math.min(1, Math.max(0, (v - 0.3) * 2))) / 3) * 3
      if (d < 3) continue
      const role = v > 0.66 ? 'p' : v > 0.55 ? 'b' : 'm'
      // A zero-length segment with a round cap is a dot of diameter stroke-width.
      layer.stroke(role, `stroke-width="${d}" stroke-linecap="round"`, `M${pt(x, y)}h0`)
    }
  }
  return String(layer)
}

const VARIANTS = [blocks, truchet, bauhaus, flow, halftone]

/**
 * SVG markup for the artwork of `slug`, `width` SVG units wide and ART_HEIGHT
 * tall. The pattern and its parameters depend only on the slug; `width` only
 * sets how far the pattern extends.
 */
export function artSvg(slug, width) {
  const seed = hashSeed(slug)
  const random = makeRandom(seed)
  const noise = makeNoise(seed)
  const height = ART_HEIGHT
  const body = VARIANTS[(seed >>> 3) % VARIANTS.length]({ width, height, random, noise })
  const grain = `pa-grain-${seed.toString(36)}-${width}`
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid slice">` +
    `<rect class="fg" width="${width}" height="${height}"/>` +
    body +
    // Print-like speckle: thresholded fractal noise as the alpha of an ink-coloured sheet.
    `<filter id="${grain}" x="0" y="0" width="1" height="1">` +
    `<feTurbulence type="fractalNoise" baseFrequency=".9" seed="${seed % 1000}" result="n"/>` +
    `<feColorMatrix in="n" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 0 0 0 -16"/>` +
    `<feComposite in="SourceGraphic" operator="in"/>` +
    `</filter>` +
    `<rect class="fi" width="${width}" height="${height}" opacity=".22" filter="url(#${grain})"/>` +
    `</svg>`
  )
}
