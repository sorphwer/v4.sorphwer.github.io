/**
 * Seeded generative artwork for post banners and related-post cards, drawn with
 * p5 (instance mode, see ArtCanvas). Composition is derived from the post slug,
 * so a post always gets the same picture and its card matches its banner; the
 * colour scheme follows the site theme.
 *
 * Palette is DESIGN.md's: Pure Black / 50% Grey / Pure White as the base,
 * one RS Blue per theme (RS Blue Light on the light theme's white ground,
 * RS Blue Dark on the dark theme's black ground) and Warning Pink, picked at
 * roughly 6 : 3 : 1.
 */

const BLACK = '#000000'
const GREY = '#808080'
const WHITE = '#ffffff'
const RS_BLUE_LIGHT = '#64d2ff'
const RS_BLUE_DARK = '#0070c9'
const WARNING_PINK = '#e83e8c'

/** FNV-1a: stable 32-bit seed from a slug. */
export function hashSeed(text) {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** Ground and inks for the site theme. */
function artScheme(dark) {
  return dark
    ? { ground: BLACK, ink: WHITE, mid: GREY, blue: RS_BLUE_DARK, pink: WARNING_PINK }
    : { ground: WHITE, ink: BLACK, mid: GREY, blue: RS_BLUE_LIGHT, pink: WARNING_PINK }
}

/** Base (grey / ink) : blue : pink ≈ 6 : 3 : 1 over the drawn shapes. */
function pickInk(p, s) {
  const r = p.random()
  if (r < 0.6) return r < 0.4 ? s.mid : s.ink
  if (r < 0.9) return s.blue
  return s.pink
}

/** Isometric cube field, the Notion-cover look: shaded faces, a few coloured lids. */
function cubes(p, s) {
  const a = p.height / p.random(3.2, 5.5)
  const w = a * Math.sqrt(3)
  const left = p.lerpColor(p.color(s.ground), p.color(s.mid), 0.45)
  const right = p.lerpColor(p.color(s.ground), p.color(s.mid), 0.15)
  p.noStroke()
  for (let row = -1; row * a * 1.5 < p.height + a * 2; row++) {
    for (let col = -1; col * w < p.width + w; col++) {
      if (p.random() < 0.08) continue
      const x = col * w + (row % 2 ? w / 2 : 0)
      const y = row * a * 1.5
      const r = p.random()
      const lid = r < 0.16 ? s.blue : r < 0.2 ? s.pink : s.mid
      p.fill(lid)
      p.quad(x, y - a, x + w / 2, y - a / 2, x, y, x - w / 2, y - a / 2)
      p.fill(left)
      p.quad(x - w / 2, y - a / 2, x, y, x, y + a, x - w / 2, y + a / 2)
      p.fill(right)
      p.quad(x, y, x + w / 2, y - a / 2, x + w / 2, y + a / 2, x, y + a)
    }
  }
}

/** Truchet quarter-arcs: continuous grey strands, some cells re-inked. */
function truchet(p, s) {
  const size = p.height / Math.floor(p.random(3, 6))
  p.noFill()
  p.strokeCap(p.SQUARE)
  p.strokeWeight(size * 0.2)
  for (let y = 0; y < p.height; y += size) {
    for (let x = 0; x < p.width; x += size) {
      p.stroke(p.random() < 0.55 ? s.mid : pickInk(p, s))
      if (p.random() < 0.5) {
        p.arc(x, y, size, size, 0, p.HALF_PI)
        p.arc(x + size, y + size, size, size, p.PI, p.PI + p.HALF_PI)
      } else {
        p.arc(x + size, y, size, size, p.HALF_PI, p.PI)
        p.arc(x, y + size, size, size, p.PI + p.HALF_PI, p.TWO_PI)
      }
    }
  }
}

/** Bauhaus tiles: each cell one primitive (quarter / half disc, circle, triangle, bars). */
function bauhaus(p, s) {
  const size = p.height / Math.floor(p.random(2, 4))
  p.noStroke()
  for (let y = 0; y < p.height; y += size) {
    for (let x = 0; x < p.width; x += size) {
      const back = p.random() < 0.55 ? s.ground : pickInk(p, s)
      let front = pickInk(p, s)
      if (front === back) front = back === s.ground ? s.mid : s.ground
      p.fill(back)
      p.rect(x, y, size, size)
      p.fill(front)
      const turn = Math.floor(p.random(4)) * p.HALF_PI
      const kind = Math.floor(p.random(6))
      p.push()
      p.translate(x + size / 2, y + size / 2)
      p.rotate(turn)
      if (kind === 0) p.arc(-size / 2, -size / 2, size * 2, size * 2, 0, p.HALF_PI, p.PIE)
      else if (kind === 1) p.arc(0, size / 2, size, size, p.PI, p.TWO_PI, p.PIE)
      else if (kind === 2) p.circle(0, 0, size * 0.62)
      else if (kind === 3)
        p.triangle(-size / 2, -size / 2, size / 2, -size / 2, -size / 2, size / 2)
      else if (kind === 4) {
        for (let i = 0; i < 3; i++) p.rect(-size / 2, -size / 2 + (i * size) / 3, size, size / 7)
      }
      p.pop()
    }
  }
}

/** Perlin flow field: thin grey strands with blue and pink currents. */
function flow(p, s) {
  const k = p.random(0.002, 0.005)
  const twist = p.random(1.5, 3)
  const count = Math.floor((p.width * p.height) / 900)
  p.noFill()
  for (let i = 0; i < count; i++) {
    let x = p.random(-40, p.width + 40)
    let y = p.random(-40, p.height + 40)
    const c = p.color(pickInk(p, s))
    c.setAlpha(p.random(120, 230))
    p.stroke(c)
    p.strokeWeight(p.random(0.6, 2.2))
    p.beginShape()
    const steps = Math.floor(p.random(20, 70))
    for (let j = 0; j < steps; j++) {
      p.vertex(x, y)
      const angle = p.noise(x * k, y * k) * p.TWO_PI * twist
      x += Math.cos(angle) * 4
      y += Math.sin(angle) * 4
    }
    p.endShape()
  }
}

/** Halftone: a dot grid whose radii follow a noise wave; coloured bands ride the crests. */
function halftone(p, s) {
  const step = p.height / p.random(14, 24)
  const k = p.random(0.004, 0.01)
  p.noStroke()
  for (let y = step / 2; y < p.height; y += step) {
    for (let x = step / 2; x < p.width; x += step) {
      const n = p.noise(x * k, y * k)
      const d = step * 0.95 * p.constrain((n - 0.3) * 2, 0, 1)
      if (d < 1) continue
      p.fill(n > 0.68 ? s.pink : n > 0.55 ? s.blue : s.mid)
      p.circle(x, y, d)
    }
  }
}

const VARIANTS = [cubes, truchet, bauhaus, flow, halftone]

/** Print-like speckle over the whole artwork. */
function grain(p, s) {
  const count = Math.floor((p.width * p.height) / 45)
  const light = p.color(s.ink)
  const dark = p.color(s.ground)
  light.setAlpha(28)
  dark.setAlpha(60)
  p.strokeWeight(1)
  for (let i = 0; i < count; i++) {
    p.stroke(i % 2 ? light : dark)
    p.point(p.random(p.width), p.random(p.height))
  }
}

/**
 * Draw the artwork for `seed` onto the whole p5 canvas in the light or dark
 * scheme. The seed alone fixes the composition, so both themes show the same
 * picture with swapped ground and inks.
 */
export function drawArt(p, seed, dark) {
  const scheme = artScheme(dark)
  p.randomSeed(seed)
  p.noiseSeed(seed)
  p.background(scheme.ground)
  VARIANTS[(seed >>> 3) % VARIANTS.length](p, scheme)
  grain(p, scheme)
}
