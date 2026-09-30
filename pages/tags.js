import { useMemo, useState } from 'react'
import Link from '@/components/Link'
import { PageSEO } from '@/components/SEO'
import SourceMark from '@/components/home/SourceMark'
import { SearchIcon } from '@/components/home/icons'
import TagBubbles from '@/components/tags/TagBubbles'
import TagDetail from '@/components/tags/TagDetail'
import TagTimeline from '@/components/tags/TagTimeline'
import siteMetadata from '@/data/siteMetadata'
import { getAllFilesFrontMatter } from '@/lib/mdx'
import { buildTagAtlas } from '@/lib/tagAtlas'

export async function getStaticProps() {
  const posts = (await getAllFilesFrontMatter('blog')).map((post) => ({
    date: post.date,
    tags: post.tags,
    source: post.notion ? 'notion' : 'mdx',
  }))
  return { props: { atlas: buildTagAtlas(posts), postCount: posts.length } }
}

const byLabel = (a, b) => a.label.localeCompare(b.label, 'en', { sensitivity: 'base' })
const initialOf = (tag) => {
  const c = tag.label[0].toUpperCase()
  return c >= 'A' && c <= 'Z' ? c : '#'
}

function SectionTitle({ children, note }) {
  return (
    <div className="mb-4">
      <h2 className="text-lg font-bold tracking-tight text-gray-900 dark:text-gray-100">
        {children}
      </h2>
      {note && <p className="mt-0.5 text-sm text-gray-500 dark:text-gray-400">{note}</p>}
    </div>
  )
}

/** Every tag (filtered by `matches`) in letter groups laid out in columns. */
function TagIndex({ tags, matches }) {
  const groups = new Map()
  for (const tag of [...tags].sort(byLabel)) {
    if (matches && !matches.has(tag.key)) continue
    const letter = initialOf(tag)
    if (!groups.has(letter)) groups.set(letter, [])
    groups.get(letter).push(tag)
  }
  return (
    <div className="gap-8 sm:columns-2 lg:columns-3 xl:columns-4">
      {[...groups].map(([letter, group]) => (
        <section key={letter} className="mb-5 break-inside-avoid">
          <h3 className="mb-1 border-b border-gray-200 pb-1 text-xs font-semibold text-gray-900 dark:border-gray-800 dark:text-gray-100">
            {letter}
          </h3>
          <ul className="space-y-0.5">
            {group.map((tag) => (
              <li key={tag.key}>
                <Link
                  href={`/tags/${tag.key}`}
                  className="text-sm text-gray-600 hover:text-primary-600 dark:text-gray-400 dark:hover:text-primary-400"
                >
                  {tag.label}
                  <span className="ml-1.5 text-xs tabular-nums text-gray-400 dark:text-gray-500">
                    {tag.count}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}

/**
 * Tag atlas: a packed bubble chart of every topic (size = posts, colour =
 * recent or earlier) with a detail panel for the hovered tag, a timeline of
 * the recurring topics by year, and an A–Z index. The filter box dims
 * non-matching tags in the charts and narrows the index. Clicking a tag opens
 * /tags/<slug>. Layout is computed at build time (lib/tagAtlas).
 */
export default function Tags({ atlas, postCount }) {
  const { tags, years, postsPerYear, recentFrom, sources } = atlas
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(null)
  const [showIndex, setShowIndex] = useState(false)

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (needle === '') return null
    return new Set(
      tags
        .filter((tag) => tag.label.toLowerCase().includes(needle) || tag.key.includes(needle))
        .map((tag) => tag.key)
    )
  }, [tags, query])
  // With a filter that leaves one tag, show it in the panel.
  const single = matches && matches.size === 1 ? [...matches][0] : null
  const activeTag = tags.find((tag) => tag.key === (active ?? single)) ?? null
  const recurring = tags.filter((tag) => tag.count > 1).length

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
        {matches && (
          <p aria-live="polite" className="text-sm text-gray-500 dark:text-gray-400">
            {matches.size} {matches.size === 1 ? 'match' : 'matches'}
          </p>
        )}
      </div>

      <section className="grid gap-8 border-b border-gray-200 pb-12 dark:border-gray-800 lg:grid-cols-[minmax(0,1fr)_17rem] lg:items-center">
        <div className="mx-auto w-full max-w-[40rem]">
          <TagBubbles
            tags={tags}
            recentFrom={recentFrom}
            active={activeTag?.key ?? null}
            matches={matches}
            onActive={setActive}
          />
        </div>
        <aside
          aria-live="polite"
          className="lg:border-l lg:border-gray-200 lg:pl-8 lg:dark:border-gray-800"
        >
          <TagDetail
            tag={activeTag}
            years={years}
            postsPerYear={postsPerYear}
            recentFrom={recentFrom}
            tagCount={tags.length}
          />
        </aside>
      </section>

      <section className="border-b border-gray-200 py-12 dark:border-gray-800">
        <SectionTitle
          note={`The ${recurring} tags used in more than one post, in order of first appearance. Dot size = posts that year.`}
        >
          Topics over time
        </SectionTitle>
        <TagTimeline
          tags={tags}
          years={years}
          postsPerYear={postsPerYear}
          recentFrom={recentFrom}
          active={activeTag?.key ?? null}
          matches={matches}
          onActive={setActive}
        />
      </section>

      <section className="py-12">
        <button
          type="button"
          aria-expanded={showIndex || !!matches}
          onClick={() => setShowIndex(!showIndex)}
          className="text-left"
        >
          <SectionTitle note="Every tag, alphabetically, with its post count.">
            All tags A–Z <span className="text-gray-400">{showIndex || matches ? '−' : '+'}</span>
          </SectionTitle>
        </button>
        {(showIndex || matches) && <TagIndex tags={tags} matches={matches} />}
      </section>
    </>
  )
}
