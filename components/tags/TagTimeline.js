import Link from '@/components/Link'
import { filterHref } from '@/lib/utils/postFacets'

/**
 * Dot matrix of the recurring tags (more than one post) against years: dot
 * area ∝ posts that year, blue when the tag is still written about
 * (`recentFrom`), grey otherwise. Rows run in order of first appearance, so
 * the chart reads as the site's topics shifting over time. Each row links to
 * the home page filtered to the tag; rows outside `matches` are dimmed.
 */
export default function TagTimeline({ tags, years, postsPerYear, recentFrom, matches }) {
  const rows = tags
    .filter((tag) => tag.count > 1)
    .sort((a, b) => a.first - b.first || b.count - a.count || a.label.localeCompare(b.label))
  const peak = Math.max(...rows.flatMap((tag) => tag.perYear))
  const columns = {
    gridTemplateColumns: `minmax(8rem, 12rem) repeat(${years.length}, minmax(2rem, 1fr)) 3rem`,
  }

  return (
    <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
      <div className="min-w-[36rem]">
        <div
          style={columns}
          className="grid items-end border-b border-gray-200 pb-2 text-[11px] tabular-nums text-gray-400 dark:border-gray-800 dark:text-gray-500"
        >
          <span />
          {years.map((year, i) => (
            <span key={year} className="text-center" title={`${postsPerYear[i]} posts`}>
              {year}
            </span>
          ))}
          <span className="pr-2 text-right">posts</span>
        </div>
        <ul>
          {rows.map((tag) => {
            const dimmed = matches && !matches.has(tag.key)
            const dot = tag.last >= recentFrom ? 'bg-primary-500' : 'bg-gray-400 dark:bg-gray-500'
            return (
              <li
                key={tag.key}
                className={`transition-opacity duration-200 ${dimmed ? 'opacity-20' : ''}`}
              >
                <Link
                  href={filterHref(tag.key)}
                  aria-label={`${tag.label}: ${tag.count} posts, ${tag.first}–${tag.last}`}
                  style={columns}
                  className="group grid h-8 items-center rounded-md hover:bg-gray-100 focus-visible:bg-gray-100 focus-visible:outline-none dark:hover:bg-gray-800/60 dark:focus-visible:bg-gray-800/60"
                >
                  <span className="truncate pl-2 text-sm text-gray-700 group-hover:text-primary-600 dark:text-gray-300 dark:group-hover:text-primary-400">
                    {tag.label}
                  </span>
                  {tag.perYear.map((n, i) => {
                    const size = 6 + 12 * Math.sqrt(n / peak)
                    return (
                      <span
                        key={years[i]}
                        className="flex justify-center"
                        title={n ? `${years[i]}: ${n}` : undefined}
                      >
                        {n > 0 ? (
                          <span
                            className={`rounded-full ${dot} group-hover:bg-RSpink`}
                            style={{ width: size, height: size }}
                          />
                        ) : (
                          <span className="h-px w-3 bg-gray-200 dark:bg-gray-800" />
                        )}
                      </span>
                    )
                  })}
                  <span className="pr-2 text-right text-xs tabular-nums text-gray-400 dark:text-gray-500">
                    {tag.count}
                  </span>
                </Link>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}
