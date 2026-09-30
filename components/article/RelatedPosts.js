import Link from '@/components/Link'
import siteMetadata from '@/data/siteMetadata'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import ArtCanvas from './ArtCanvas'

const cardDate = { year: 'numeric', month: 'short', day: 'numeric' }

function CardLabel({ kind, label }) {
  if (kind === 'prev') return <span>&larr; {label}</span>
  if (kind === 'next') return <span>{label} &rarr;</span>
  return (
    <span>
      <FontAwesomeIcon icon="tags" className="mr-1.5" />
      {label}
    </span>
  )
}

/**
 * "Keep reading": previous / next and tag-related posts as cards, each topped
 * with that post's own banner artwork. Cards come from lib/utils/relatedPosts.
 */
export default function RelatedPosts({ posts }) {
  return (
    <section
      aria-labelledby="keep-reading"
      className="mt-10 border-t border-gray-200 pt-10 dark:border-gray-800"
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2
          id="keep-reading"
          className="font-rs text-2xl font-medium tracking-tight text-gray-900 dark:text-gray-100"
        >
          Keep reading
        </h2>
        <Link
          href="/blog"
          className="text-sm font-medium text-primary-500 hover:text-primary-600 dark:text-primary-400 dark:hover:text-primary-400"
        >
          All posts &rarr;
        </Link>
      </div>
      {posts.length > 0 && (
        <ul className="mt-6 grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
          {posts.map((post) => (
            <li key={post.slug}>
              <Link
                href={`/blog/${post.slug}`}
                className="group flex h-full flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white transition duration-200 hover:-translate-y-0.5 hover:border-gray-300 hover:shadow-lg dark:border-gray-800 dark:bg-gray-800/40 dark:hover:border-gray-700"
              >
                <ArtCanvas seed={post.slug} className="h-40 shrink-0" />
                <div className="flex flex-1 flex-col px-6 pt-5 pb-6">
                  {post.date && (
                    <time
                      dateTime={post.date}
                      className="mb-3 text-xs text-gray-500 dark:text-gray-400"
                    >
                      {new Date(post.date).toLocaleDateString(siteMetadata.locale, cardDate)}
                    </time>
                  )}
                  <h3 className="font-rs text-lg font-medium leading-snug text-gray-900 line-clamp-4 group-hover:text-primary-500 dark:text-gray-100 dark:group-hover:text-primary-400">
                    {post.title}
                  </h3>
                  <div className="mt-auto pt-6 text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                    <CardLabel kind={post.kind} label={post.label} />
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
