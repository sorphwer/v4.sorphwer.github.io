import { hierarchy, pack } from 'd3-hierarchy'
import kebabCase from './utils/kebabCase'
import { postFacets } from './utils/postFacets'
import { topicTags } from './utils/relatedPosts'

/** Side of the square the bubble chart is packed into (SVG user units). */
export const ATLAS_SIZE = 640
/** A tag counts as current when its latest post is at most this many years before the newest post. */
const RECENT_SPAN = 2

const round = (v) => Math.round(v * 10) / 10

/**
 * Everything /tags draws, computed at build time so the page ships static SVG:
 * the topic tags (grouped by slug like the home filter, labelled with their
 * latest spelling) with post counts per year and a circle-packing position
 * (area ∝ post count), the year axis with posts per year, and the source
 * (MDX / Notion) counts. `posts` need `date`, `tags` and `source`.
 */
export function buildTagAtlas(posts) {
  const { tags, sources } = postFacets(posts)
  const yearsSeen = posts.map((post) => Number(post.date.slice(0, 4)))
  const firstYear = Math.min(...yearsSeen)
  const lastYear = Math.max(...yearsSeen)
  const years = Array.from({ length: lastYear - firstYear + 1 }, (_, i) => firstYear + i)
  const yearIndex = (post) => Number(post.date.slice(0, 4)) - firstYear

  const postsPerYear = years.map(() => 0)
  const perYear = new Map(tags.map((tag) => [tag.key, years.map(() => 0)]))
  for (const post of posts) {
    const i = yearIndex(post)
    postsPerYear[i] += 1
    for (const key of new Set(topicTags(post).map(kebabCase))) perYear.get(key)[i] += 1
  }

  const root = pack().size([ATLAS_SIZE, ATLAS_SIZE]).padding(3)(
    hierarchy({ children: tags })
      .sum((node) => node.count ?? 0)
      .sort((a, b) => b.value - a.value)
  )
  const place = new Map(root.leaves().map((leaf) => [leaf.data.key, leaf]))

  const recentFrom = lastYear - RECENT_SPAN
  return {
    years,
    postsPerYear,
    recentFrom,
    sources,
    tags: tags.map((tag) => {
      const counts = perYear.get(tag.key)
      const first = years[counts.findIndex((n) => n > 0)]
      const last = years[counts.length - 1 - [...counts].reverse().findIndex((n) => n > 0)]
      const { x, y, r } = place.get(tag.key)
      return { ...tag, perYear: counts, first, last, x: round(x), y: round(y), r: round(r) }
    }),
  }
}
