import Link from '@/components/Link'
import SeasonIcon from '@/components/SeasonIcon'
import Tag from '@/components/Tag'
import formatDate from '@/lib/utils/formatDate'

/** List view entry: date column, then title, tags and summary. Tag chips call `onTagClick`. */
export default function PostRow({ post, onTagClick }) {
  const { slug, date, title, subtitle, status, summary, tags } = post
  return (
    <article className="space-y-2 xl:grid xl:grid-cols-4 xl:items-baseline xl:space-y-0">
      <dl>
        <dt className="sr-only">Published on</dt>
        <dd className="text-base font-medium leading-6 text-gray-500 dark:text-gray-400">
          <SeasonIcon date={date} />
          <time dateTime={date}>{' ' + formatDate(date)}</time>
        </dd>
      </dl>
      <div className="space-y-5 xl:col-span-3">
        <div className="space-y-6">
          <div>
            <h2 className="text-2xl font-bold leading-8 tracking-tight">
              <Link
                href={`/blog/${slug}`}
                className="text-gray-900 hover:text-primary-600 dark:text-gray-100 dark:hover:text-primary-400"
              >
                {title}
                {status && (
                  <span className="align-top text-sm font-normal text-RSpink">
                    {' [' + status + ']'}
                  </span>
                )}
                {subtitle && <div className="text-xl font-normal text-gray-500">{subtitle}</div>}
              </Link>
            </h2>
            <div className="flex flex-wrap">
              {tags.map((tag) => (
                <Tag key={tag} text={tag} onClick={onTagClick} />
              ))}
            </div>
          </div>
          {summary && (
            <div className="prose max-w-none text-gray-500 dark:text-gray-400">{summary}</div>
          )}
        </div>
        <div className="text-base font-medium leading-6">
          <Link
            href={`/blog/${slug}`}
            className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
            aria-label={`Read "${title}"`}
          >
            Read more &rarr;
          </Link>
        </div>
      </div>
    </article>
  )
}
