import Link from '@/components/Link'

/**
 * Dot matrix of the recurring tags (more than one post) against years: dot
 * area ∝ posts that year. Rows run in order of first appearance, so the
 * chart reads as the site's topics shifting over time. Rows outside `matches`
 * are dimmed; hovering a row reports it through `onActive`.
 */
export default function TagTimeline({
  tags,
  years,
  postsPerYear,
  recentFrom,
  active,
  matches,
  onActive,
}) {
  const rows = tags
    .filter((tag) => tag.count > 1)
    .sort((a, b) => a.first - b.first || b.count - a.count || a.label.localeCompare(b.label))
  const peak = Math.max(...rows.flatMap((tag) => tag.perYear))
  const columns = {
    gridTemplateColumns: `minmax(8rem, 12rem) repeat(${years.length}, minmax(2rem, 1fr)) 2.5rem`,
  }

  return (
    <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
      <div className="min-w-[36rem]" onMouseLeave={() => onActive(null)}>
        <div
          style={columns}
          className="grid items-end border-b border-gray-200 pb-2 text-[11px] tabular-nums text-gray-400 dark:border-gray-800 dark:text-gray-500"
        >
          <span />
          {years.map((year, i) => (
            <span key={year} className="text-center" title={`${postsPerYear[i]} posts`}>
              {`’${String(year).slice(2)}`}
            </span>
          ))}
          <span className="text-right">posts</span>
        </div>
        {rows.map((tag) => {
          const isActive = active === tag.key
          const dimmed = matches && !matches.has(tag.key)
          const dot = isActive
            ? 'bg-RSpink'
            : tag.last >= recentFrom
            ? 'bg-primary-500'
            : 'bg-gray-400 dark:bg-gray-500'
          return (
            <div
              key={tag.key}
              style={columns}
              onMouseEnter={() => onActive(tag.key)}
              className={`grid h-7 items-center rounded-md transition-opacity duration-200 ${
                isActive ? 'bg-gray-100 dark:bg-gray-800/60' : ''
              } ${dimmed ? 'opacity-20' : ''}`}
            >
              <Link
                href={`/tags/${tag.key}`}
                onFocus={() => onActive(tag.key)}
                className="truncate pl-2 text-sm text-gray-700 hover:text-primary-600 dark:text-gray-300 dark:hover:text-primary-400"
              >
                {tag.label}
              </Link>
              {tag.perYear.map((n, i) => (
                <span
                  key={years[i]}
                  className="flex justify-center"
                  title={n ? `${years[i]}: ${n}` : undefined}
                >
                  {n > 0 ? (
                    <span
                      className={`rounded-full ${dot}`}
                      style={{
                        width: 6 + 12 * Math.sqrt(n / peak),
                        height: 6 + 12 * Math.sqrt(n / peak),
                      }}
                    />
                  ) : (
                    <span className="h-px w-3 bg-gray-200 dark:bg-gray-800" />
                  )}
                </span>
              ))}
              <span className="pr-2 text-right text-xs tabular-nums text-gray-400 dark:text-gray-500">
                {tag.count}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
