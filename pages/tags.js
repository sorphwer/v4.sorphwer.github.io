import { useMemo, useState } from 'react'
import Link from '@/components/Link'
import { PageSEO } from '@/components/SEO'
import SourceMark from '@/components/home/SourceMark'
import { SearchIcon } from '@/components/home/icons'
import siteMetadata from '@/data/siteMetadata'
import { getAllFilesFrontMatter } from '@/lib/mdx'
import { postFacets } from '@/lib/utils/postFacets'

export async function getStaticProps() {
  const posts = (await getAllFilesFrontMatter('blog')).map((post) => ({
    date: post.date,
    tags: post.tags,
    source: post.notion ? 'notion' : 'mdx',
  }))
  const { tags, sources } = postFacets(posts)
  return { props: { tags, sources, postCount: posts.length } }
}

const SORTS = [
  { key: 'popular', label: 'Popular' },
  { key: 'az', label: 'A–Z' },
]

const byLabel = (a, b) => a.label.localeCompare(b.label, 'en', { sensitivity: 'base' })
const initialOf = (tag) => {
  const c = tag.label[0].toUpperCase()
  return c >= 'A' && c <= 'Z' ? c : '#'
}

function SortToggle({ sort, onChange }) {
  return (
    <div
      role="group"
      aria-label="Sort tags"
      className="flex shrink-0 rounded-xl bg-gray-100 p-1 dark:bg-gray-800/60"
    >
      {SORTS.map(({ key, label }) => (
        <button
          key={key}
          type="button"
          aria-pressed={sort === key}
          onClick={() => onChange(key)}
          className={`rounded-lg px-3 py-1.5 text-sm transition-colors ${
            sort === key
              ? 'bg-white text-gray-900 shadow-sm ring-1 ring-gray-200 dark:bg-gray-900 dark:text-gray-100 dark:ring-gray-700'
              : 'text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

/** Bordered chip with the post count; for tags used more than once. */
function TagPill({ tag }) {
  return (
    <Link
      href={`/tags/${tag.key}`}
      className="group inline-flex items-center gap-1.5 rounded-lg border border-gray-200 px-2.5 py-1 text-sm text-gray-700 transition-colors hover:border-primary-500 hover:text-primary-600 dark:border-gray-800 dark:text-gray-300 dark:hover:border-primary-400 dark:hover:text-primary-400"
    >
      {tag.label}
      <span className="text-xs tabular-nums text-gray-400 group-hover:text-primary-500 dark:text-gray-500">
        {tag.count}
      </span>
    </Link>
  )
}

/** Plain text link; for the long tail and the A–Z index. */
function TagLink({ tag, showCount = false }) {
  return (
    <Link
      href={`/tags/${tag.key}`}
      className="text-sm text-gray-600 hover:text-primary-600 dark:text-gray-400 dark:hover:text-primary-400"
    >
      {tag.label}
      {showCount && (
        <span className="ml-1.5 text-xs tabular-nums text-gray-400 dark:text-gray-500">
          {tag.count}
        </span>
      )}
    </Link>
  )
}

function SectionTitle({ children, note }) {
  return (
    <h2 className="mb-3 flex items-baseline gap-2 text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
      {children}
      {note != null && (
        <span className="tabular-nums text-gray-400 dark:text-gray-500">{note}</span>
      )}
    </h2>
  )
}

/**
 * Tag index. Popular: tags used more than once as chips (most used first), then
 * the one-off tags as a dense alphabetical run of links. A–Z: every tag in
 * letter groups laid out in columns. The search box filters both views. Source
 * (MDX / Notion) is listed apart from the topic tags.
 */
export default function Tags({ tags, sources, postCount }) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState('popular')

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return needle === ''
      ? tags
      : tags.filter((tag) => tag.label.toLowerCase().includes(needle) || tag.key.includes(needle))
  }, [tags, query])

  const frequent = shown.filter((tag) => tag.count > 1)
  const once = shown.filter((tag) => tag.count === 1).sort(byLabel)
  const letters = useMemo(() => {
    const groups = new Map()
    for (const tag of [...shown].sort(byLabel)) {
      const letter = initialOf(tag)
      if (!groups.has(letter)) groups.set(letter, [])
      groups.get(letter).push(tag)
    }
    return [...groups]
  }, [shown])

  return (
    <>
      <PageSEO title={`Tags - ${siteMetadata.author}`} description="Things I blog about" />
      <header className="border-b border-gray-200 pt-6 pb-6 dark:border-gray-700">
        <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10">
          Tags
        </h1>
        <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-gray-500 dark:text-gray-400">
          <span>
            {tags.length} topics across {postCount} posts
          </span>
          <span aria-hidden="true">·</span>
          <span className="whitespace-nowrap">
            {sources.map((source, i) => (
              <span key={source.key}>
                {i > 0 && <span className="mx-1.5 text-gray-300 dark:text-gray-600">/</span>}
                <Link
                  href={`/tags/${source.key}`}
                  className="hover:text-primary-600 dark:hover:text-primary-400"
                >
                  <SourceMark source={source.key} />
                  <span className="ml-1 tabular-nums text-gray-400 dark:text-gray-500">
                    {source.count}
                  </span>
                </Link>
              </span>
            ))}
          </span>
        </p>
      </header>

      <div className="flex items-center gap-4 py-6">
        <label className="relative min-w-0 flex-1 sm:max-w-sm">
          <span className="sr-only">Filter tags</span>
          <SearchIcon className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter tags"
            className="block w-full rounded-xl border border-gray-200 bg-white py-2.5 pl-10 pr-4 text-sm text-gray-900 placeholder-gray-400 focus:border-primary-500 focus:ring-primary-500 dark:border-gray-800 dark:bg-gray-800/40 dark:text-gray-100 dark:placeholder-gray-500"
          />
        </label>
        <p aria-live="polite" className="hidden text-sm text-gray-500 dark:text-gray-400 sm:block">
          {shown.length} {shown.length === 1 ? 'tag' : 'tags'}
        </p>
        <div className="ml-auto">
          <SortToggle sort={sort} onChange={setSort} />
        </div>
      </div>

      {shown.length === 0 ? (
        <p className="pb-16 text-sm text-gray-500 dark:text-gray-400">
          No tags match “{query.trim()}”.
        </p>
      ) : sort === 'popular' ? (
        <div className="space-y-10 pb-16">
          {frequent.length > 0 && (
            <section>
              <SectionTitle note={frequent.length}>Used in several posts</SectionTitle>
              <div className="flex flex-wrap gap-2">
                {frequent.map((tag) => (
                  <TagPill key={tag.key} tag={tag} />
                ))}
              </div>
            </section>
          )}
          {once.length > 0 && (
            <section>
              <SectionTitle note={once.length}>Used once</SectionTitle>
              <div className="flex flex-wrap gap-x-4 gap-y-1.5">
                {once.map((tag) => (
                  <TagLink key={tag.key} tag={tag} />
                ))}
              </div>
            </section>
          )}
        </div>
      ) : (
        <div className="gap-8 pb-16 sm:columns-2 lg:columns-3 xl:columns-4">
          {letters.map(([letter, group]) => (
            <section key={letter} className="mb-6 break-inside-avoid">
              <h2 className="mb-1.5 border-b border-gray-200 pb-1 text-xs font-semibold text-gray-900 dark:border-gray-800 dark:text-gray-100">
                {letter}
              </h2>
              <ul className="space-y-0.5">
                {group.map((tag) => (
                  <li key={tag.key}>
                    <TagLink tag={tag} showCount />
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </>
  )
}
