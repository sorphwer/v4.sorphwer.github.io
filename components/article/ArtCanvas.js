import { useEffect, useRef } from 'react'
import { useTheme } from 'next-themes'
import { drawArt, hashSeed } from './art'

/**
 * Hosts one static p5 sketch (instance mode) that fills its box with the
 * seeded artwork from ./art in the current site theme. p5 is imported on the
 * client only; until it loads, the box shows the theme's ground colour.
 * Redraws on resize and on light/dark switches. Sizing comes from `className`
 * (give it a height).
 */
export default function ArtCanvas({ seed, className = '' }) {
  const ref = useRef(null)
  const sketchRef = useRef(null)
  const darkRef = useRef(false)
  const { resolvedTheme } = useTheme()
  const dark = resolvedTheme === 'dark'
  const numericSeed = hashSeed(seed)

  useEffect(() => {
    const host = ref.current
    let observer = null
    let cancelled = false

    import('p5').then(({ default: P5 }) => {
      if (cancelled) return
      // next-themes sets the class before hydration, so the first draw is already
      // in the right scheme even if `resolvedTheme` hasn't populated yet.
      darkRef.current = document.documentElement.classList.contains('dark')
      const sketch = new P5((p) => {
        p.setup = () => {
          p.createCanvas(host.clientWidth, host.clientHeight)
          p.noLoop()
        }
        p.draw = () => drawArt(p, numericSeed, darkRef.current)
      }, host)
      sketchRef.current = sketch
      observer = new ResizeObserver(() => {
        const { clientWidth, clientHeight } = host
        if (clientWidth && (clientWidth !== sketch.width || clientHeight !== sketch.height)) {
          sketch.resizeCanvas(clientWidth, clientHeight)
        }
      })
      observer.observe(host)
    })

    return () => {
      cancelled = true
      observer?.disconnect()
      sketchRef.current?.remove()
      sketchRef.current = null
    }
  }, [numericSeed])

  // `resolvedTheme` is undefined on the server and first client render; the
  // ground class covers that gap, then the canvas redraws in the real theme.
  useEffect(() => {
    if (resolvedTheme === undefined) return
    darkRef.current = dark
    sketchRef.current?.redraw()
  }, [dark, resolvedTheme])

  return (
    <div
      ref={ref}
      aria-hidden="true"
      className={`art-canvas relative overflow-hidden bg-white dark:bg-black ${className}`}
    />
  )
}
