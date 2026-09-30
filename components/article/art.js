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
 * Colours are not baked in: shapes carry role classes (`f*` fill, `s*` stroke,
 * `f* si` fill with an ink outline, `f* t5|t7` a 50 / 70% tint of the
 * fill) that css/tailwind.css maps to DESIGN.md's palette per theme. Light:
 * white ground, black ink, RS Blue Light; dark: black ground, white ink, RS
 * Blue Dark; 50% Grey and Warning Pink in both. Roles: g ground, i ink, m grey,
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

/** `count` unit directions, rounded so platform trig differences cannot leak into output. */
const directions = (count) =>
  Array.from({ length: count }, (_, i) => [
    Math.round(Math.cos((i * 2 * Math.PI) / count) * 1e6) / 1e6,
    Math.round(Math.sin((i * 2 * Math.PI) / count) * 1e6) / 1e6,
  ])
const DIRECTIONS = directions(64)
/** 1.5° steps; every petal count in `garden` divides 240. */
const WHEEL = directions(240)

const n = Math.round
const pt = (x, y) => `${n(x)} ${n(y)}`
const polygon = (points) => `M${points.map(([x, y]) => pt(x, y)).join('L')}Z`
/** Path through `points` in relative moves (deltas of the rounded points, so no drift). */
const trace = (points, close = false) => {
  const r = points.map(([x, y]) => [n(x), n(y)])
  const moves = r.slice(1).map(([x, y], i) => `${x - r[i][0]} ${y - r[i][1]}`)
  return `M${r[0][0]} ${r[0][1]}l${moves.join(' ')}${close ? 'z' : ''}`
}
const arc = (x0, y0, x1, y1, radius) =>
  `M${pt(x0, y0)}A${n(radius)} ${n(radius)} 0 0 1 ${pt(x1, y1)}`
/** Radius rounded before use, so the two arcs meet exactly and stay round. */
const circle = (cx, cy, radius) => {
  const r = n(radius)
  return `M${n(cx) - r} ${n(cy)}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`
}

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
    fill: (role, d, tint = '') => add(`f${role}${tint}`, `class="f${role}${tint}"`, d),
    stroke: (role, attrs, d) => add(`s${role}${attrs}`, `class="s${role}" ${attrs}`, d),
    outlined: (role, attrs, d, tint = '') =>
      add(`o${role}${tint}${attrs}`, `class="f${role}${tint} si" ${attrs}`, d),
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

const OUTLINE = 'stroke-width="1.5" stroke-linejoin="round"'
const PETAL_COUNTS = [8, 10, 12, 16, 20]
/** Tint classes (`t5` = 50% of the colour); 0 keeps the plain colour. */
const TINTS = [0, 5, 7]

/** Flower head colour: blue, ink or pink; never grey or ground. */
function pickHead(random) {
  const r = random()
  if (r < 0.55) return 'b'
  if (r < 0.8) return 'i'
  return 'p'
}

/** A shape of `role`: blue, pink and grey take a random tint; ground and ink stay plain. */
function tinted(random, role, d) {
  const tint = role === 'g' || role === 'i' ? 0 : TINTS[random.int(TINTS.length)]
  return { role, d, tint }
}

/**
 * One plant rooted at (x, ground): a straight stem, kite leaves alternating up
 * it, and a pinwheel head of skewed triangular petals around a disc. Returns
 * the stem path and the plant's shapes with their horizontal extent.
 */
function makePlant(x, ground, height, random) {
  const radius = random.range(20, 38)
  const hx = x + random.range(-10, 10)
  const hy = random.range(radius + 4, height * 0.6)
  const plant = {
    z: random(),
    lo: hx - radius - 1,
    hi: hx + radius + 1,
    stem: trace([
      [x, ground],
      [hx, hy],
    ]),
    leaves: [],
    veins: '',
  }

  // Leaves point 12°–54° above horizontal; veins run from the midrib toward the tip.
  // They start a head radius below the head, so none peeks through a petal gap as a scrap.
  const span = ground - hy - radius * 1.2
  const count = Math.max(2, Math.min(6, Math.round(span / 38 + random())))
  let side = random() < 0.5 ? 1 : -1
  for (let j = 0; j < count; j++) {
    const lift = ((j + random.range(0.15, 0.85)) / count) * span
    const sx = x + ((hx - x) * lift) / (ground - hy)
    const sy = ground - lift
    const [c, s] = WHEEL[side > 0 ? 232 - random.int(29) : 128 + random.int(29)]
    const length = random.range(22, 46)
    const half = length * random.range(0.22, 0.3)
    const at = (u, v) => [sx + u * c - v * s, sy + u * s + v * c]
    const bends = [random.range(0.3, 0.6), random.range(0.3, 0.6)]
    const points = [
      at(1, 0),
      at(length * bends[0], half),
      at(length, 0),
      at(length * bends[1], -half),
    ]
    for (const [px] of points) {
      plant.lo = Math.min(plant.lo, px - 1)
      plant.hi = Math.max(plant.hi, px + 1)
    }
    plant.leaves.push(tinted(random, random() < 0.6 ? 'm' : 'g', trace(points, true)))
    plant.veins += trace([at(1, 0), at(length * 0.85, 0)])
    const ribs = length > 32 ? 3 : 2
    bends.forEach((bend, k) => {
      const sign = k ? -1 : 1
      for (let r = 1; r <= ribs; r++) {
        const u = length * (0.12 + (0.76 * r) / (ribs + 1))
        const v =
          u < length * bend
            ? (half * u) / (length * bend)
            : (half * (length - u)) / (length * (1 - bend))
        const from = u - v * 0.9
        if (from > 2) plant.veins += trace([at(from, 0), at(u, sign * v)])
      }
    })
    side = -side
  }

  // Petal k: apex near the centre, outer corners at radius and radius · skew, so the head spins.
  const role = pickHead(random)
  const petals = PETAL_COUNTS[random.int(PETAL_COUNTS.length)]
  const slot = 240 / petals
  const spread = Math.max(3, Math.round(slot * random.range(0.55, 0.9)))
  const skew = random.range(0.65, 1)
  const [lead, trail] = random() < 0.5 ? [1, skew] : [skew, 1]
  const alternate = random() < 0.3 ? random.range(0.75, 0.9) : 1
  const start = random.int(240)
  let head = ''
  for (let k = 0; k < petals; k++) {
    const a = start + k * slot
    const r = radius * (k % 2 ? alternate : 1)
    const at = (i, rr) => {
      const [c, s] = WHEEL[(a + i) % 240]
      return [hx + c * rr, hy + s * rr]
    }
    head += trace([at(spread >> 1, radius * 0.12), at(0, r * lead), at(spread, r * trail)], true)
  }
  plant.head = tinted(random, role, head)
  const disc = role === 'p' || (role === 'i' && random() < 0.4) ? 'g' : 'p'
  plant.disc = tinted(random, disc, circle(hx, hy, Math.max(3, radius * 0.22)))
  return plant
}

/** Blue and pink go unoutlined: an ink ring on colour reads badly, worst in dark. */
function paint(layer, { role, d, tint }) {
  const cls = tint ? ` t${tint}` : ''
  if (role === 'b' || role === 'p') layer.fill(role, d, cls)
  else layer.outlined(role, OUTLINE, d, cls)
}

/**
 * Pinwheel garden, after a p5 sketch: plants along a ground line, blue / ink /
 * pink heads, grey and ground leaves outlined in ink, tints for variety.
 * Stems and the ground line sit behind everything.
 * Plants overlap, so they are painted in a random depth order: each joins the
 * batch above the highest earlier plant it overlaps, so a batch holds disjoint
 * plants and draws as a few style paths (leaves, veins, petals, discs).
 */
function garden({ width, height, random }) {
  const ground = height - 12
  const stems = makeLayer()
  stems.stroke('i', 'stroke-width="1.5"', `M0 ${ground}H${width}`)
  const plants = []
  for (let x = random.range(0, 40); x < width + 40; x += random.range(42, 80)) {
    const plant = makePlant(x, ground, height, random)
    stems.stroke('i', 'stroke-width="1.5"', plant.stem)
    plants.push(plant)
  }
  plants.sort((a, b) => a.z - b.z)
  const batches = []
  plants.forEach((plant, i) => {
    let level = 0
    for (let j = 0; j < i; j++) {
      const other = plants[j]
      if (other.lo < plant.hi && plant.lo < other.hi) level = Math.max(level, other.level + 1)
    }
    plant.level = level
    if (!batches[level]) batches[level] = Array.from({ length: 4 }, makeLayer)
    const [leaves, veins, petals, discs] = batches[level]
    for (const leaf of plant.leaves) paint(leaves, leaf)
    veins.stroke('i', 'stroke-width=".75"', plant.veins)
    paint(petals, plant.head)
    paint(discs, plant.disc)
  })
  return String(stems) + batches.map((layers) => layers.join('')).join('')
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

const VARIANTS = [blocks, truchet, garden, flow, halftone]

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
