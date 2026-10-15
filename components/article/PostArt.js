import { useMemo } from 'react'
import { artSvg } from './art'

/** Banner width in SVG units: wide enough that large screens only scale, never repeat. */
export const BANNER_ART_WIDTH = 2400
/** Card width in SVG units: cards show a centred crop, so a narrower field is enough. */
export const CARD_ART_WIDTH = 720

/**
 * Seeded post artwork (./art) as inline SVG. It is plain markup computed during
 * render, so it is in the statically generated HTML and needs no client script;
 * colours follow the site theme through CSS (css/tailwind.css `.post-art`).
 * The SVG covers the box (`slice`), so sizing comes from `className`.
 */
export default function PostArt({ seed, width = BANNER_ART_WIDTH, className = '' }) {
  const markup = useMemo(() => artSvg(seed, width), [seed, width])
  return (
    <div
      aria-hidden="true"
      className={`post-art overflow-hidden ${className}`}
      dangerouslySetInnerHTML={{ __html: markup }}
    />
  )
}
