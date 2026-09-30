import NextLink from 'next/link'
import { ATLAS_SIZE } from '@/lib/tagAtlas'

// Estimated advance of an Inter glyph, as a fraction of the font size.
const GLYPH = 0.58

/** Label size that fits inside the circle, or 0 when even the smallest size would overflow. */
function labelSize(tag) {
  const size = Math.min(15, tag.r * 0.42, (1.7 * tag.r) / (GLYPH * tag.label.length))
  return size >= 8 ? size : 0
}

/**
 * Packed bubble chart of every topic tag: area ∝ post count, blue when the tag
 * was written about recently (`recent`), grey otherwise, pink when active.
 * Each bubble links to the tag page; hover / focus reports the tag through
 * `onActive`. Tags outside `matches` (the filter) are dimmed.
 */
export default function TagBubbles({ tags, recentFrom, active, matches, onActive }) {
  return (
    <svg
      viewBox={`0 0 ${ATLAS_SIZE} ${ATLAS_SIZE}`}
      className="h-auto w-full select-none"
      role="list"
      aria-label="Tags sized by number of posts"
      onMouseLeave={() => onActive(null)}
    >
      {tags.map((tag) => {
        const size = labelSize(tag)
        const isActive = active === tag.key
        const recent = tag.last >= recentFrom
        const dimmed = matches && !matches.has(tag.key)
        const fill = isActive
          ? 'fill-RSpink'
          : recent
          ? 'fill-primary-500'
          : 'fill-gray-200 dark:fill-gray-800'
        const ink = isActive || recent ? 'fill-white' : 'fill-gray-700 dark:fill-gray-300'
        return (
          <NextLink key={tag.key} href={`/tags/${tag.key}`} passHref>
            <a
              role="listitem"
              aria-label={`${tag.label}, ${tag.count} post${tag.count === 1 ? '' : 's'}`}
              onMouseEnter={() => onActive(tag.key)}
              onFocus={() => onActive(tag.key)}
              className="outline-none"
            >
              <g
                className={`transition-opacity duration-200 ${dimmed ? 'opacity-15' : ''}`}
                transform={`translate(${tag.x} ${tag.y})`}
              >
                <circle r={tag.r} className={`${fill} transition-colors duration-150`} />
                {size > 0 && (
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    y={tag.r >= 30 ? -size * 0.45 : 0}
                    fontSize={size}
                    className={`${ink} pointer-events-none font-medium`}
                  >
                    {tag.label}
                  </text>
                )}
                {size > 0 && tag.r >= 30 && (
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    y={size * 0.85}
                    fontSize={size * 0.8}
                    className={`${ink} pointer-events-none tabular-nums opacity-70`}
                  >
                    {tag.count}
                  </text>
                )}
              </g>
            </a>
          </NextLink>
        )
      })}
    </svg>
  )
}
