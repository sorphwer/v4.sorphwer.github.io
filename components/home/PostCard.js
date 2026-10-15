import Link from '@/components/Link'
import SeasonIcon from '@/components/SeasonIcon'
import Tag from '@/components/Tag'
import PostArt, { CARD_ART_WIDTH } from '@/components/article/PostArt'
import formatDate, { SHORT_DATE } from '@/lib/utils/formatDate'
import kebabCase from '@/lib/utils/kebabCase'

/** Chips on a card: the source chip (MDX / Notion) plus two topic tags. */
const CARD_TAGS = 3

/**
 * Grid view card: the post's banner art cropped as cover, date, title and tag
 * chips. The title link is stretched over the whole card; the chips sit above
 * it and call `onTagClick` instead of opening the post. Tags in `activeTags`
 * (filter keys) come first, so a filtered card always shows the chip that
 * turns its filter off.
 */
export default function PostCard({ post, activeTags, onTagClick }) {
  const { slug, date, title, subtitle, status, tags } = post
  const [source, ...topics] = tags
  const isActive = (tag) => activeTags.includes(kebabCase(tag))
  // Array#sort is stable: active tags first, each group in its original order.
  const chips = [source, ...topics.sort((a, b) => isActive(b) - isActive(a))].slice(0, CARD_TAGS)
  return (
    <article className="group relative flex h-full flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white transition duration-200 hover:-translate-y-0.5 hover:border-gray-300 hover:shadow-lg dark:border-gray-800 dark:bg-gray-800/40 dark:hover:border-gray-700">
      <PostArt
        seed={slug}
        width={CARD_ART_WIDTH}
        className="aspect-[16/10] shrink-0 border-b border-gray-200 dark:border-gray-800"
      />
      <div className="flex flex-1 flex-col px-6 pt-5 pb-6">
        <p className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
          <SeasonIcon date={date} />
          <time dateTime={date}>{formatDate(date, SHORT_DATE)}</time>
          {status && <span className="text-RSpink">[{status}]</span>}
        </p>
        <h2 className="mt-3 font-rs text-lg font-medium leading-snug text-gray-900 line-clamp-3 group-hover:text-primary-500 dark:text-gray-100 dark:group-hover:text-primary-400">
          <Link href={`/blog/${slug}`} className="after:absolute after:inset-0">
            {title}
          </Link>
        </h2>
        {subtitle && (
          <p className="mt-1.5 text-sm text-gray-500 line-clamp-2 dark:text-gray-400">{subtitle}</p>
        )}
        <div className="pointer-events-none relative mt-auto flex flex-wrap items-center pt-5">
          {chips.map((tag) => (
            <Tag
              key={tag}
              text={tag}
              onClick={onTagClick}
              className="pointer-events-auto max-w-full truncate"
            />
          ))}
        </div>
      </div>
    </article>
  )
}
