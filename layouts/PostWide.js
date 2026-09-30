import PostArticle from '@/components/article/PostArticle'

/**
 * Layout for posts whose body is one large interactive React component
 * (paper, glance). Same frame as PostLayout (PostArticle) but no `prose`
 * wrapper: the post owns its typography, aligned with the site's prose in
 * css/glance.css and components/posts/paper. `bodyClass` frontmatter scopes
 * the post's stylesheet (`paper`, `glance`). Notion is unsupported here.
 */
export default function PostWide({ frontMatter, authorDetails, related, children }) {
  return (
    <PostArticle
      frontMatter={frontMatter}
      authorDetails={authorDetails}
      title={frontMatter.title}
      related={related}
    >
      <div className={frontMatter.bodyClass || undefined}>{children}</div>
    </PostArticle>
  )
}
