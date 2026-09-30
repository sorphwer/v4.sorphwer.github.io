import PostArticle from '@/components/article/PostArticle'

/**
 * Default post layout: Markdown/MDX prose, or a Notion page (`notion`
 * frontmatter) rendered by MDXLayoutRenderer into `NotionJsx`.
 * The banner, header, heading rail and related posts come from PostArticle.
 */
export default function PostLayout({
  frontMatter,
  authorDetails,
  related,
  NotionJsx,
  NotionTitle,
  children,
}) {
  return (
    <PostArticle
      frontMatter={frontMatter}
      authorDetails={authorDetails}
      title={NotionTitle || frontMatter.title}
      related={related}
    >
      <div className="prose max-w-none dark:prose-dark">
        {children}
        {NotionJsx}
      </div>
    </PostArticle>
  )
}
