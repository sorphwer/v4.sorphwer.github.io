import { useEffect, useMemo, useRef, useState } from 'react'
import Router, { useRouter } from 'next/router'
import kebabCase from '@/lib/utils/kebabCase'
import {
  DEFAULT_FILTERS,
  SOURCES,
  filterPosts,
  filtersFromSearch,
  filtersToSearch,
  postFacets,
} from '@/lib/utils/postFacets'
import FilterPanel from './FilterPanel'
import PostCard from './PostCard'
import PostRow from './PostRow'
import { ChevronIcon, GridIcon, ListIcon, SearchIcon } from './icons'

/** Posts rendered per "Show more" step; each card carries an inline SVG cover. */
const PAGE_SIZE = 12
const VIEW_KEY = 'home-view'
const VIEWS = [
  { key: 'grid', label: 'Grid', Icon: GridIcon },
  { key: 'list', label: 'List', Icon: ListIcon },
]
const NO_FACETS = { years: [], sources: [], tags: [] }

function ViewToggle({ view, onChange }) {
  return (
    <div
      role="group"
      aria-label="Layout"
      className="flex shrink-0 rounded-xl bg-gray-100 p-1 dark:bg-gray-800/60"
    >
      {VIEWS.map(({ key, label, Icon }) => (
        <button
          key={key}
          type="button"
          aria-pressed={view === key}
          aria-label={label}
          onClick={() => onChange(key)}
          className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition-colors ${
            view === key
              ? 'bg-white text-gray-900 shadow-sm ring-1 ring-gray-200 dark:bg-gray-900 dark:text-gray-100 dark:ring-gray-700'
              : 'text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100'
          }`}
        >
          <Icon className="h-4 w-4" />
          <span className="hidden sm:inline">{label}</span>
        </button>
      ))}
    </div>
  )
}

/**
 * Home page post browser: "Filter and sort" sidebar (sort, year, source, tag),
 * search, and a grid / list toggle. Filters live in the URL query
 * (lib/utils/postFacets filtersToSearch), so a filtered view can be linked;
 * the chosen view is remembered in localStorage. `posts` must be newest first.
 */
export default function PostBrowser({ posts }) {
  const facets = useMemo(() => postFacets(posts), [posts])
  const { isReady } = useRouter()
  const [filters, setFilters] = useState(DEFAULT_FILTERS)
  const [limit, setLimit] = useState(PAGE_SIZE)
  const [view, setView] = useState('grid')
  const [filtersOpen, setFiltersOpen] = useState(false)
  const resultsRef = useRef(null)
  // Set once the URL has been read, so the empty initial state never overwrites it.
  const urlRead = useRef(false)

  useEffect(() => {
    if (localStorage.getItem(VIEW_KEY) === 'list') setView('list')
  }, [])

  // State → URL, as a shallow replace (no history entry per click or keystroke).
  // Declared before the URL → state effect, so on mount it bails out before the URL is read.
  useEffect(() => {
    if (!urlRead.current || !isReady) return
    const search = filtersToSearch(filters)
    if (search === window.location.search.slice(1)) return
    Router.replace(search ? `${Router.pathname}?${search}` : Router.pathname, undefined, {
      shallow: true,
      scroll: false,
    })
  }, [filters, isReady])

  // URL → state: on load, and when a link (not our own shallow replace) lands on this page.
  useEffect(() => {
    const apply = (search) => {
      setFilters(filtersFromSearch(search, facets))
      setLimit(PAGE_SIZE)
    }
    apply(window.location.search)
    urlRead.current = true
    const onRouteChange = (url, { shallow }) => {
      if (!shallow) apply(new URL(url, window.location.origin).search)
    }
    Router.events.on('routeChangeComplete', onRouteChange)
    return () => Router.events.off('routeChangeComplete', onRouteChange)
  }, [facets])

  const results = useMemo(() => filterPosts(posts, filters), [posts, filters])
  const shown = results.slice(0, limit)
  const activeFacets = filters.years.length + filters.sources.length + filters.tags.length

  const update = (change) => {
    setFilters(change)
    setLimit(PAGE_SIZE)
  }
  const toggle = (field, key) =>
    update((current) => ({
      ...current,
      [field]: current[field].includes(key)
        ? current[field].filter((k) => k !== key)
        : [...current[field], key],
    }))
  // A tag chip toggles its filter: MDX / Notion chips the source filter, the rest the tag filter.
  const toggleTag = (tag) => {
    const key = kebabCase(tag)
    toggle(SOURCES.some((source) => source.key === key) ? 'sources' : 'tags', key)
    // The result list changes; bring its top back into view if the chip was far down.
    if (resultsRef.current.getBoundingClientRect().top < 0) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }
  const chooseView = (next) => {
    setView(next)
    localStorage.setItem(VIEW_KEY, next)
  }

  return (
    <div className="xl:grid xl:grid-cols-[13rem_minmax(0,1fr)] xl:gap-12">
      {/* On xl the sidebar is sticky and capped at the viewport height; only FilterPanel's
          tag list shrinks and scrolls. overflow-y-auto is a fallback for very short windows;
          overflow-x-hidden keeps the chevron's rotation (whose box briefly exceeds the
          edge mid-transition) from flashing a horizontal scrollbar. pr-3 is the lane
          FilterPanel's list scrollbars sit in, right of the content column. */}
      <aside className="mb-8 pr-3 xl:sticky xl:top-8 xl:mb-0 xl:flex xl:max-h-[calc(100vh-4rem)] xl:flex-col xl:self-start xl:overflow-y-auto xl:overflow-x-hidden">
        <div className="flex items-center justify-between gap-4 pb-3">
          <h2 className="hidden font-rs text-base font-medium text-gray-900 dark:text-gray-100 xl:block">
            Filter and sort
          </h2>
          <button
            type="button"
            aria-expanded={filtersOpen}
            aria-controls="post-filters"
            onClick={() => setFiltersOpen(!filtersOpen)}
            className="flex items-center gap-2 font-rs text-base font-medium text-gray-900 dark:text-gray-100 xl:hidden"
          >
            Filter and sort
            <ChevronIcon
              className={`h-4 w-4 transition-transform ${filtersOpen ? 'rotate-180' : ''}`}
            />
          </button>
          {activeFacets > 0 && (
            <button
              type="button"
              onClick={() => update((current) => ({ ...current, ...NO_FACETS }))}
              className="text-xs text-primary-500 hover:underline dark:text-primary-400"
            >
              Clear ({activeFacets})
            </button>
          )}
        </div>
        <div
          id="post-filters"
          className={`${filtersOpen ? '' : 'hidden'} xl:flex xl:min-h-0 xl:flex-col`}
        >
          <FilterPanel
            facets={facets}
            filters={filters}
            onToggle={toggle}
            onSort={(sort) => update((current) => ({ ...current, sort }))}
          />
        </div>
      </aside>

      <section ref={resultsRef} aria-label="Posts" className="min-w-0 scroll-mt-8">
        <div className="flex items-center gap-4">
          <label className="relative min-w-0 flex-1 sm:max-w-sm">
            <span className="sr-only">Search posts</span>
            <SearchIcon className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              type="search"
              value={filters.query}
              onChange={(e) => {
                const query = e.target.value
                update((current) => ({ ...current, query }))
              }}
              placeholder="Search posts"
              className="block w-full rounded-xl border border-gray-200 bg-white py-2.5 pl-10 pr-4 text-sm text-gray-900 placeholder-gray-400 focus:border-primary-500 focus:ring-primary-500 dark:border-gray-800 dark:bg-gray-800/40 dark:text-gray-100 dark:placeholder-gray-500"
            />
          </label>
          <p
            aria-live="polite"
            className="hidden text-sm text-gray-500 dark:text-gray-400 sm:block"
          >
            {results.length} {results.length === 1 ? 'post' : 'posts'}
          </p>
          <div className="ml-auto">
            <ViewToggle view={view} onChange={chooseView} />
          </div>
        </div>

        {results.length === 0 ? (
          <div className="mt-16 text-center text-gray-500 dark:text-gray-400">
            <p>No posts match these filters.</p>
            <button
              type="button"
              onClick={() => update((current) => ({ ...current, ...NO_FACETS, query: '' }))}
              className="mt-3 text-sm text-primary-500 hover:underline dark:text-primary-400"
            >
              Clear search and filters
            </button>
          </div>
        ) : view === 'grid' ? (
          <ul className="mt-8 grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
            {shown.map((post) => (
              <li key={post.slug}>
                <PostCard post={post} activeTags={filters.tags} onTagClick={toggleTag} />
              </li>
            ))}
          </ul>
        ) : (
          <ul className="mt-4 divide-y divide-gray-200 dark:divide-gray-700">
            {shown.map((post) => (
              <li key={post.slug} className="py-10">
                <PostRow post={post} onTagClick={toggleTag} />
              </li>
            ))}
          </ul>
        )}

        {results.length > limit && (
          <div className="mt-10 flex justify-center">
            <button
              type="button"
              onClick={() => setLimit(limit + PAGE_SIZE)}
              className="rounded-xl border border-gray-200 px-5 py-2 text-sm text-gray-700 transition-colors hover:border-gray-400 hover:text-gray-900 dark:border-gray-800 dark:text-gray-300 dark:hover:border-gray-600 dark:hover:text-gray-100"
            >
              Show more
              <span className="ml-2 text-gray-400 dark:text-gray-500">
                {results.length - limit} left
              </span>
            </button>
          </div>
        )}
      </section>
    </div>
  )
}
