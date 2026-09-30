import { useRef } from 'react'
import Link from '@/components/Link'
import PageTitle from '@/components/PageTitle'
import SectionContainer from '@/components/SectionContainer'
import { BlogSEO } from '@/components/SEO'
import Tag from '@/components/Tag'
import ScrollTopAndComment from '@/components/ScrollTopAndComment'
import siteMetadata from '@/data/siteMetadata'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import ArtCanvas from './ArtCanvas'
import PostToc from './PostToc'
import RelatedPosts from './RelatedPosts'
import { LangProvider, LangToggle, T } from './lang'

const editUrl = (fileName) => `${siteMetadata.siteRepo}/blob/master/data/blog/${fileName}`

const postDateTemplate = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' }

/**
 * Shared frame for every blog post (MDX prose, Notion, component-built posts):
 *
 *   full-bleed seeded p5 banner
 *   sheet overlapping the banner: title + byline | language toggle + tags
 *   rule
 *   body (`children`, the layout's own wrapper) with the fixed heading rail
 *   "Keep reading" cards (previous / next / tag-related)
 *
 * `title` is the resolved title (Notion pages carry their own). A `titleZh`
 * frontmatter makes the post bilingual: the post is wrapped in LangProvider,
 * the title follows the toggle, and the body reads the language via `T`/`useT`.
 */
export default function PostArticle({ frontMatter, authorDetails, title, related, children }) {
  const { slug, fileName, date, tags, titleZh, notion } = frontMatter
  const bodyRef = useRef(null)
  const bilingual = Boolean(titleZh)

  const article = (
    <SectionContainer>
      <BlogSEO
        url={`${siteMetadata.siteUrl}/blog/${slug}`}
        authorDetails={authorDetails}
        {...frontMatter}
      />
      <ScrollTopAndComment />
      <article>
        {/* Full-bleed: centred in the viewport regardless of the container's width. */}
        <ArtCanvas
          seed={slug}
          className="ml-[calc(50%-50vw)] h-44 w-screen border-y border-gray-200 dark:border-gray-800 sm:h-56 lg:h-64"
        />
        {/* The ::before strip carries the shadow so it only falls on the banner overlap. */}
        <div className="relative z-0 -mx-4 -mt-10 rounded-t-2xl bg-white px-4 pt-8 before:absolute before:inset-x-0 before:top-0 before:-z-10 before:h-10 before:rounded-t-2xl before:shadow-[0_-8px_24px_-12px_rgba(0,0,0,0.35)] dark:bg-gray-900 sm:mx-auto sm:-mt-16 sm:max-w-[60rem] sm:px-12 sm:pt-10 sm:before:h-16">
          <header className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-start">
            <div className="min-w-0 space-y-3">
              <PageTitle>{bilingual ? <T en={title} zh={titleZh} /> : title}</PageTitle>
              <div className="flex flex-wrap items-center gap-x-2 text-base font-medium leading-6 text-gray-600 dark:text-gray-300">
                <span className="font-rs italic">
                  {authorDetails.map((author) => author.name).join(', ')}
                </span>
                <span aria-hidden="true">·</span>
                <time dateTime={date}>
                  {new Date(date).toLocaleDateString(siteMetadata.locale, postDateTemplate)}
                </time>
                {!notion && (
                  <Link
                    href={editUrl(fileName)}
                    aria-label="Edit on GitHub"
                    className="ml-1 text-sm text-gray-400 hover:text-primary-600 dark:text-gray-500 dark:hover:text-primary-400"
                  >
                    <FontAwesomeIcon icon="edit" />
                  </Link>
                )}
              </div>
            </div>
            {(bilingual || tags?.length > 0) && (
              <div className="flex flex-col gap-3 lg:max-w-xs lg:items-end lg:pt-2">
                {bilingual && <LangToggle />}
                {tags?.length > 0 && (
                  <div className="flex flex-wrap lg:-mr-3 lg:justify-end">
                    {tags.map((tag) => (
                      <Tag key={tag} text={tag} />
                    ))}
                  </div>
                )}
              </div>
            )}
          </header>
          <div className="mt-8 h-px bg-gray-200 dark:bg-gray-700" />
          <div ref={bodyRef} className="pt-8 pb-8">
            {children}
          </div>
        </div>
        <PostToc rootRef={bodyRef} />
        <RelatedPosts posts={related ?? []} />
      </article>
    </SectionContainer>
  )

  return bilingual ? <LangProvider>{article}</LangProvider> : article
}
