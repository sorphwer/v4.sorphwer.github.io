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
 * b blue, p pink. The seascape's halftone plates take their colour from
 * `k*` classes on feFlood, and its theme-dependent tones from `t*` classes on
 * gradient stops, so dark mode is CSS-only there too.
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

/** Petal colour: ink, blue or pink; never grey or ground. */
function pickPetal(random) {
  const r = random()
  if (r < 0.5) return 'i'
  if (r < 0.85) return 'b'
  return 'p'
}

/**
 * Pinwheel of curved petals around (cx, cy), one of `counts` many: each runs
 * from the centre out along a leading edge swept back, round a bulging rim, and
 * straight home. Outer corners sit at radius and radius · skew, so the wheel
 * spins; the rim reaches at most 1.15 · radius.
 */
function pinwheel(cx, cy, radius, counts, random) {
  const count = counts[random.int(counts.length)]
  const slot = 240 / count
  const spread = Math.max(3, Math.round(slot * random.range(0.55, 0.95)))
  const skew = random.range(0.6, 1)
  const [lead, trail] = random() < 0.5 ? [1, skew] : [skew, 1]
  const start = random.int(240)
  let d = ''
  for (let k = 0; k < count; k++) {
    const at = (i, r) => {
      const [c, s] = WHEEL[(start + k * slot + i + 240) % 240]
      return [cx + c * r, cy + s * r]
    }
    const sweep = at(-(spread >> 1), radius * 0.5)
    const tip = at(0, radius * lead)
    const rim = at(spread >> 1, (radius * (lead + trail) * 1.3) / 2)
    const tail = at(spread, radius * trail)
    d += `M${pt(cx, cy)}Q${pt(...sweep)} ${pt(...tip)}Q${pt(...rim)} ${pt(...tail)}Z`
  }
  return d
}

/**
 * One plant rising from below the bottom edge: a bowed stem, drooping almond
 * leaves alternating up it, and a pinwheel head around a disc. Returns the stem
 * path and the plant's shapes with their horizontal extent.
 */
function makePlant(x, height, random) {
  const radius = random.range(30, 54)
  const hx = x + random.range(-12, 12)
  const hy = random.range(radius * 0.5, height * 0.55)
  // The stem's control point sits at mid-height, so its y is linear in t.
  const y0 = height + 4
  const mx = (x + hx) / 2 + random.range(-20, 20)
  const stemX = (t) => (1 - t) * (1 - t) * x + 2 * t * (1 - t) * mx + t * t * hx
  const role = pickPetal(random)
  const plant = {
    z: random(),
    lo: hx - radius * 1.15,
    hi: hx + radius * 1.15,
    stem: `M${pt(x, y0)}Q${pt(mx, (y0 + hy) / 2)} ${pt(hx, hy)}`,
    leaves: [],
    head: [role, pinwheel(hx, hy, radius, [6, 8, 10, 12], random)],
    disc: [role !== 'p' && random() < 0.4 ? 'p' : 'g', circle(hx, hy, radius * 0.2)],
  }
  // Leaves point 12°–54° above horizontal, starting a head radius below the head.
  const span = height - hy - radius
  const count = Math.max(2, Math.min(5, Math.round(span / 40 + random())))
  let side = random() < 0.5 ? 1 : -1
  for (let j = 0; j < count; j++) {
    const lift = ((j + random.range(0.15, 0.85)) / count) * span
    const sx = stemX((lift + 4) / (y0 - hy))
    const sy = height - lift
    const [c, s] = WHEEL[side > 0 ? 232 - random.int(29) : 128 + random.int(29)]
    const length = random.range(28, 56)
    const half = length * random.range(0.25, 0.32)
    const at = (u, v) => [sx + u * c - v * s, sy + u * s + v * c]
    const bends = [random.range(0.3, 0.6), random.range(0.3, 0.6)]
    // Almond: two quadratic sides around an axis whose tip droops; a side
    // bulges by half its control offset. The curve stays inside these points.
    const droop = side * length * random.range(0.05, 0.2)
    const points = [
      at(0, 0),
      at(length * bends[0], 1.6 * half + droop / 2),
      at(length, droop),
      at(length * bends[1], droop / 2 - 1.6 * half),
    ]
    for (const [px] of points) {
      plant.lo = Math.min(plant.lo, px)
      plant.hi = Math.max(plant.hi, px)
    }
    const [base, left, tip, right] = points.map(([px, py]) => pt(px, py))
    plant.leaves.push([random() < 0.8 ? 'm' : 'i', `M${base}Q${left} ${tip}Q${right} ${base}Z`])
    side = -side
  }
  return plant
}

/**
 * Pinwheel garden, after a p5 sketch, drawn flat like the other patterns:
 * bowed grey stems run off the bottom edge, grey / ink almond leaves, ink /
 * blue / pink heads overlapping into a canopy, no outlines. Stems sit behind
 * everything. Plants overlap, so they are painted in a random depth order:
 * each joins the batch above the highest earlier plant it overlaps, so a
 * batch holds disjoint plants and draws as a few style paths.
 */
function garden({ width, height, random }) {
  const stems = makeLayer()
  const plants = []
  for (let x = random.range(0, 40); x < width + 40; x += random.range(36, 70)) {
    const plant = makePlant(x, height, random)
    stems.stroke('m', 'stroke-width="4"', plant.stem)
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
    if (!batches[level]) batches[level] = Array.from({ length: 3 }, makeLayer)
    const [leaves, petals, discs] = batches[level]
    for (const [role, d] of plant.leaves) leaves.fill(role, d)
    petals.fill(...plant.head)
    discs.fill(...plant.disc)
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

/** Seascape halftone: plate role → square dot lattice direction (≈ 18°, 45°, 72°), back to front. */
const SEA_SCREENS = [
  ['b', 3, 1],
  ['m', 1, 1],
  ['p', 1, 3],
]
/** Seascape dot pitch in SVG units. */
const SEA_PITCH = 6

/**
 * One halftone cell tile as a data URI: a square dot lattice along (a, b),
 * periodic in an L×L tile (L integer), so feTile repeats it without rotating
 * the filter region. Each dot is a radial "hill" (opaque centre, fading out
 * past the cell's half-diagonal) that the plate filter thresholds against
 * coverage, so dot area grows with coverage.
 */
function screenTile(a, b) {
  const k = a * a + b * b
  const root = Math.sqrt(k)
  const L = n(SEA_PITCH * root)
  const step = L / k
  const pitch = L / root
  const r = (pitch * 0.6).toFixed(2)
  const dots = []
  for (let i = -k; i <= 2 * k; i++) {
    for (let j = -k; j <= 2 * k; j++) {
      const x = (i * a - j * b) * step
      const y = (i * b + j * a) * step
      if (x > -pitch && x < L + pitch && y > -pitch && y < L + pitch) {
        dots.push(`<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r}" fill="url(#g)"/>`)
      }
    }
  }
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${L}" height="${L}"><radialGradient id="g">` +
    `<stop offset="0"/><stop offset=".5" stop-opacity=".75"/><stop offset=".75" stop-opacity=".44"/><stop offset="1" stop-opacity="0"/>` +
    `</radialGradient>${dots.join('')}</svg>`
  return { L, uri: `data:image/svg+xml,${encodeURIComponent(svg)}` }
}

/**
 * Seascape, a tonal halftone: coverage per colour role comes from gradients
 * and feTurbulence fields (cloud band, ripples); a filter per role turns that
 * coverage into a dot screen in the browser, each plate slightly out of
 * register. Pale sky with a soft cloud band, a low sun with pink haze and a
 * broken glint on a rippled sea. In dark mode the CSS dims the inks and moves
 * the light to the horizon (css/tailwind.css).
 */
function seascape({ width, height, random, seed }) {
  const ns = `pa-sea-${seed.toString(36)}-${width}-`
  const box = `x="0" y="0" width="${width}" height="${height}"`
  const area = `maskUnits="userSpaceOnUse" ${box}`
  /** Vertical gradient; `stops` are offset, opacity pairs, stop k classed `${cls}${k}` for theming. */
  const vgrad = (id, y1, y2, cls, ...stops) => {
    let markup = `<linearGradient id="${ns}${id}" gradientUnits="userSpaceOnUse" x1="0" y1="${y1}" x2="0" y2="${y2}">`
    for (let k = 0; k < stops.length; k += 2) {
      markup += `<stop class="${cls}${k / 2}" offset="${stops[k]}" stop-opacity="${stops[k + 1]}"/>`
    }
    return `${markup}</linearGradient>`
  }
  // Noise as alpha (gain · noise + bias), white for luminance masks or black.
  const noiseFilter = (id, fx, fy, octaves, noiseSeed, gain, bias, white) =>
    `<filter id="${ns}${id}" filterUnits="userSpaceOnUse" ${box} color-interpolation-filters="sRGB">` +
    `<feTurbulence type="fractalNoise" baseFrequency="${fx} ${fy}" numOctaves="${octaves}" seed="${noiseSeed}"/>` +
    `<feColorMatrix type="matrix" values="0 0 0 0 ${white} 0 0 0 0 ${white} 0 0 0 0 ${white} ${gain} 0 0 0 ${bias}"/>` +
    `</filter>`

  const horizon = 150 + n(random() * 30)
  const sunX = n(width * (0.4 + random() * 0.2))
  const sunR = 18 + n(random() * 8)
  const sunY = horizon - n(sunR * (0.2 + random() * 0.9))
  const sea = height - horizon
  const water = horizon + 3
  const sunHole = sunR + 2
  const hazeR = n(width * 0.22)
  const glintX = sunX - 70
  const glintEnd = sunX + 70
  const glintY = horizon + 2
  const cloudH = horizon - 6
  const defs =
    vgrad('sky', 0, horizon, 'tsky', 0, 0.34, 0.7, 0.12, 1, 0.04) +
    vgrad('sea', horizon, height, 'tsea', 0, 0.42, 0.12, 0.3, 1, 0.55) +
    `<linearGradient id="${ns}band" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="${horizon}">` +
    `<stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".3" stop-color="#fff"/><stop offset=".8" stop-color="#fff" stop-opacity=".7"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>` +
    `<radialGradient id="${ns}haze" gradientUnits="userSpaceOnUse" cx="${sunX}" cy="${horizon}" r="${hazeR}" gradientTransform="translate(0 ${horizon}) scale(1 .28) translate(0 -${horizon})">` +
    `<stop offset="0" stop-opacity=".26"/><stop offset="1" stop-opacity="0"/></radialGradient>` +
    `<linearGradient id="${ns}glint" gradientUnits="userSpaceOnUse" x1="${glintX}" y1="0" x2="${glintEnd}" y2="0">` +
    `<stop offset="0" stop-opacity="0"/><stop offset=".5" stop-opacity=".6"/><stop offset="1" stop-opacity="0"/></linearGradient>` +
    noiseFilter('cloud', 0.0014, 0.012, 4, seed % 997, 5, -2.45, 1) +
    noiseFilter('ripple', 0.0035, 0.09, 3, (seed >>> 10) % 997, 2.4, -0.9, 0) +
    noiseFilter('ripple-w', 0.0035, 0.09, 3, (seed >>> 10) % 997, 2.4, -0.9, 1) +
    // Cloud band: noise kept to the upper sky by a vertical gradient.
    `<mask id="${ns}band-m" ${area}><rect width="${width}" height="${horizon}" fill="url(#${ns}band)"/></mask>` +
    `<mask id="${ns}cloud-m" ${area}><g mask="url(#${ns}band-m)"><rect width="${width}" height="${horizon}" filter="url(#${ns}cloud)"/></g></mask>` +
    // Sky minus clouds and the sun; sea minus ripple highlights; glint only on ripples.
    `<mask id="${ns}sky-m" ${area}><rect width="${width}" height="${height}" fill="#fff"/>` +
    `<g mask="url(#${ns}cloud-m)"><rect width="${width}" height="${horizon}"/></g><circle cx="${sunX}" cy="${sunY}" r="${sunHole}"/></mask>` +
    `<mask id="${ns}sea-m" ${area}><rect width="${width}" height="${height}" fill="#fff"/>` +
    `<rect y="${water}" width="${width}" height="${sea}" filter="url(#${ns}ripple)"/></mask>` +
    `<mask id="${ns}glint-m" ${area}><rect y="${water}" width="${width}" height="${sea}" filter="url(#${ns}ripple-w)"/></mask>` +
    `<clipPath id="${ns}above"><rect width="${width}" height="${horizon}"/></clipPath>`
  const plates = {
    b:
      `<rect mask="url(#${ns}sky-m)" width="${width}" height="${horizon}" fill="url(#${ns}sky)"/>` +
      `<rect mask="url(#${ns}sea-m)" y="${horizon}" width="${width}" height="${sea}" fill="url(#${ns}sea)"/>`,
    // Cloud undersides: the cloud field nudged down, faint.
    m: `<g class="tcloud" mask="url(#${ns}cloud-m)" transform="translate(0 6)" opacity=".28"><rect width="${width}" height="${cloudH}"/></g>`,
    p:
      `<rect width="${width}" height="${horizon}" fill="url(#${ns}haze)"/>` +
      `<circle clip-path="url(#${ns}above)" cx="${sunX}" cy="${sunY}" r="${sunR}" fill-opacity=".5"/>` +
      `<rect x="${glintX}" y="${glintY}" width="140" height="${sea}" fill="url(#${ns}glint)" mask="url(#${ns}glint-m)"/>`,
  }

  let filters = ''
  let layers = ''
  for (const [role, a, b] of SEA_SCREENS) {
    const { L, uri } = screenTile(a, b)
    const dx = n((random() - 0.5) * 3)
    const dy = n((random() - 0.5) * 3)
    // Blurred coverage + dot hill − ½, thresholded steeply: a dot wherever the sum clears ½.
    filters +=
      `<filter id="${ns}${role}" filterUnits="userSpaceOnUse" ${box} color-interpolation-filters="sRGB">` +
      `<feImage href="${uri}" x="0" y="0" width="${L}" height="${L}" result="c"/>` +
      `<feTile in="c" result="s"/>` +
      `<feGaussianBlur in="SourceAlpha" stdDeviation=".8" result="a"/>` +
      `<feComposite in="a" in2="s" operator="arithmetic" k2="1" k3="1" k4="-.5"/>` +
      `<feComponentTransfer result="d"><feFuncA type="linear" slope="14" intercept="-6.5"/></feComponentTransfer>` +
      `<feFlood class="k${role}"/>` +
      `<feComposite in2="d" operator="in"/>` +
      `<feOffset dx="${dx}" dy="${dy}"/>` +
      `</filter>`
    layers += `<g filter="url(#${ns}${role})">${plates[role]}</g>`
  }
  return `<defs>${defs}${filters}</defs>${layers}`
}

const VARIANTS = [blocks, truchet, garden, flow, halftone, seascape]

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
  const body = VARIANTS[(seed >>> 3) % VARIANTS.length]({ width, height, random, noise, seed })
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid slice">` +
    `<rect class="fg" width="${width}" height="${height}"/>` +
    body +
    `</svg>`
  )
}
