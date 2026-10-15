import kebabCase from './utils/kebabCase'
import { postFacets } from './utils/postFacets'
import { topicTags } from './utils/relatedPosts'

/** A tag counts as current when its latest post is at most this many years before the newest post. */
const RECENT_SPAN = 2

/**
 * Data for /tags: the topic tags (grouped by slug like the home filter,
 * labelled with their latest spelling) with post counts per year and first /
 * last year, the year axis with posts per year, and the source (MDX / Notion)
 * counts. `posts` need `date`, `tags` and `source`.
 */
export function buildTagTimeline(posts) {
  const { tags, sources } = postFacets(posts)
  const yearOf = (post) => Number(post.date.slice(0, 4))
  const firstYear = Math.min(...posts.map(yearOf))
  const lastYear = Math.max(...posts.map(yearOf))
  const years = Array.from({ length: lastYear - firstYear + 1 }, (_, i) => firstYear + i)

  const postsPerYear = years.map(() => 0)
  const perYear = new Map(tags.map((tag) => [tag.key, years.map(() => 0)]))
  for (const post of posts) {
    const i = yearOf(post) - firstYear
    postsPerYear[i] += 1
    for (const key of new Set(topicTags(post).map(kebabCase))) perYear.get(key)[i] += 1
  }

  return {
    years,
    postsPerYear,
    recentFrom: lastYear - RECENT_SPAN,
    sources,
    tags: tags.map((tag) => {
      const counts = perYear.get(tag.key)
      const first = years[counts.findIndex((n) => n > 0)]
      const last = years[counts.length - 1 - [...counts].reverse().findIndex((n) => n > 0)]
      return { ...tag, perYear: counts, first, last }
    }),
  }
}
