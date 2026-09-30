import kebabCase from './kebabCase'
import { topicTags } from './relatedPosts'

/** Where a post's body comes from: a local Markdown/MDX file or a Notion page. */
export const SOURCES = [
  { key: 'mdx', label: 'MDX' },
  { key: 'notion', label: 'Notion' },
]

export const SORTS = [
  { key: 'newest', label: 'Newest first' },
  { key: 'oldest', label: 'Oldest first' },
]

/** Filter state for filterPosts; year / source / tag hold facet keys in selection order. */
export const DEFAULT_FILTERS = { query: '', sort: 'newest', years: [], sources: [], tags: [] }

/**
 * Home page link filtered to one tag chip: the MDX / Notion chips filter by
 * source (`/?source=mdx`), any other tag by tag slug (`/?tags=python`).
 */
export function filterHref(tag) {
  const key = kebabCase(tag)
  return SOURCES.some((source) => source.key === key) ? `/?source=${key}` : `/?tags=${key}`
}

// Post dates are ISO strings (lib/mdx getFileDate).
const yearOf = (post) => post.date.slice(0, 4)

// Tags are grouped by slug, so "Python" and "python" are one tag.
const tagKeys = (post) => topicTags(post).map(kebabCase)

const byCount = (a, b) => b.count - a.count || a.label.localeCompare(b.label)

/** Filter options with post counts: years (newest first), sources, tags (most used first). */
export function postFacets(posts) {
  const years = new Map()
  const sources = new Map()
  const tags = new Map()
  for (const post of posts) {
    const year = yearOf(post)
    years.set(year, (years.get(year) ?? 0) + 1)
    sources.set(post.source, (sources.get(post.source) ?? 0) + 1)
    for (const tag of new Set(topicTags(post))) {
      const key = kebabCase(tag)
      // Posts arrive newest first, so a tag is labelled with its most recent spelling.
      const entry = tags.get(key) ?? { key, label: tag, count: 0 }
      entry.count += 1
      tags.set(key, entry)
    }
  }
  return {
    years: [...years]
      .map(([key, count]) => ({ key, label: key, count }))
      .sort((a, b) => b.key.localeCompare(a.key)),
    sources: SOURCES.filter(({ key }) => sources.has(key)).map((source) => ({
      ...source,
      count: sources.get(source.key),
    })),
    tags: [...tags.values()].sort(byCount),
  }
}

const matchesQuery = (post, query) =>
  [post.title, post.subtitle, post.summary, ...post.tags]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
    .includes(query)

/**
 * Posts matching every active filter; within one filter any selected option
 * matches. `posts` must be newest first; `sort: 'oldest'` reverses the result.
 */
export function filterPosts(posts, { query, years, sources, tags, sort }) {
  const needle = query.trim().toLowerCase()
  const result = posts.filter(
    (post) =>
      (years.length === 0 || years.includes(yearOf(post))) &&
      (sources.length === 0 || sources.includes(post.source)) &&
      (tags.length === 0 || tagKeys(post).some((key) => tags.includes(key))) &&
      (needle === '' || matchesQuery(post, needle))
  )
  return sort === 'oldest' ? result.reverse() : result
}

// Filter field → URL parameter for the facet lists; values are joined with spaces ("+" in the URL).
const LIST_PARAMS = [
  ['years', 'year'],
  ['sources', 'source'],
  ['tags', 'tags'],
]

/**
 * Filters as a URL query string without "?", e.g. `q=graph&sort=oldest&year=2026+2025&source=notion&tags=python+ai`.
 * Defaults are left out, so the unfiltered page is plain `/`.
 */
export function filtersToSearch(filters) {
  const params = new URLSearchParams()
  const query = filters.query.trim()
  if (query) params.set('q', query)
  if (filters.sort !== DEFAULT_FILTERS.sort) params.set('sort', filters.sort)
  for (const [field, name] of LIST_PARAMS) {
    if (filters[field].length > 0) params.set(name, filters[field].join(' '))
  }
  return params.toString()
}

/**
 * Filters from a URL query string (with or without "?"). List values may be
 * separated by spaces or commas; tags match by slug, so `tags=Python` works.
 * Values that are not options of `facets` (see postFacets) are dropped.
 */
export function filtersFromSearch(search, facets) {
  const params = new URLSearchParams(search)
  const list = (name, options, normalize = (value) => value) => {
    const allowed = new Set(options.map((option) => option.key))
    const values = (params.get(name) ?? '')
      .split(/[\s,]+/)
      .filter(Boolean)
      .map(normalize)
    return [...new Set(values)].filter((key) => allowed.has(key))
  }
  const sort = params.get('sort')
  return {
    query: params.get('q') ?? '',
    sort: SORTS.some((option) => option.key === sort) ? sort : DEFAULT_FILTERS.sort,
    years: list('year', facets.years),
    sources: list('source', facets.sources, (value) => value.toLowerCase()),
    tags: list('tags', facets.tags, kebabCase),
  }
}
