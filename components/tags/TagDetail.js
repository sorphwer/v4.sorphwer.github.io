import Link from '@/components/Link'

/** Column chart of posts per year on the shared year axis; `max` scales the bars. */
export function YearBars({ years, counts, max, highlight = 'bg-primary-500', height = 'h-16' }) {
  return (
    <div>
      <div className={`flex items-end gap-1 ${height}`}>
        {counts.map((n, i) => (
          <div
            key={years[i]}
            title={`${years[i]}: ${n} post${n === 1 ? '' : 's'}`}
            className="flex h-full flex-1 items-end"
          >
            <div
              className={`w-full rounded-sm ${n > 0 ? highlight : 'bg-gray-100 dark:bg-gray-800'}`}
              style={{ height: n > 0 ? `${Math.max(10, (n / max) * 100)}%` : '2px' }}
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] tabular-nums text-gray-400 dark:text-gray-500">
        <span>{years[0]}</span>
        <span>{years[years.length - 1]}</span>
      </div>
    </div>
  )
}

function Legend({ recentFrom }) {
  return (
    <ul className="space-y-1.5 text-xs text-gray-500 dark:text-gray-400">
      <li className="flex items-center gap-2">
        <span className="h-3 w-3 rounded-full bg-primary-500" />
        Written about since {recentFrom}
      </li>
      <li className="flex items-center gap-2">
        <span className="h-3 w-3 rounded-full bg-gray-200 dark:bg-gray-800" />
        Earlier topics
      </li>
      <li className="flex items-center gap-2">
        <span className="flex h-3 w-3 items-center justify-center">
          <span className="h-1.5 w-1.5 rounded-full bg-gray-400" />
        </span>
        <span>
          Size <span aria-hidden="true">∝</span> number of posts
        </span>
      </li>
    </ul>
  )
}

/**
 * Side panel of the bubble chart: the active tag (name, post count, years,
 * posts per year, link), or with none active the overview and legend.
 */
export default function TagDetail({ tag, years, postsPerYear, recentFrom, tagCount }) {
  if (!tag) {
    const max = Math.max(...postsPerYear)
    return (
      <div className="space-y-6">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
            Overview
          </p>
          <p className="mt-2 text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
            {tagCount} topics
          </p>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Hover a bubble to see when it was written about; click to open its posts.
          </p>
        </div>
        <div>
          <p className="mb-2 text-xs text-gray-500 dark:text-gray-400">Posts per year</p>
          <YearBars
            years={years}
            counts={postsPerYear}
            max={max}
            highlight="bg-gray-400 dark:bg-gray-600"
          />
        </div>
        <Legend recentFrom={recentFrom} />
      </div>
    )
  }

  const span = tag.first === tag.last ? `${tag.first}` : `${tag.first}–${tag.last}`
  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
          Tag
        </p>
        <p className="mt-2 break-words text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
          {tag.label}
        </p>
        <p className="mt-1 text-sm tabular-nums text-gray-500 dark:text-gray-400">
          {tag.count} post{tag.count === 1 ? '' : 's'} · {span}
        </p>
      </div>
      <div>
        <p className="mb-2 text-xs text-gray-500 dark:text-gray-400">Posts per year</p>
        <YearBars
          years={years}
          counts={tag.perYear}
          max={Math.max(...tag.perYear)}
          highlight={tag.last >= recentFrom ? 'bg-primary-500' : 'bg-gray-500'}
        />
      </div>
      <Link
        href={`/tags/${tag.key}`}
        className="inline-flex text-sm font-medium text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
      >
        View {tag.count} post{tag.count === 1 ? '' : 's'} &rarr;
      </Link>
    </div>
  )
}
