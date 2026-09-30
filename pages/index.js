import { PageSEO } from '@/components/SEO'
import NewsletterForm from '@/components/NewsletterForm'
import PostBrowser from '@/components/home/PostBrowser'
import siteMetadata from '@/data/siteMetadata'
import { getAllFilesFrontMatter } from '@/lib/mdx'

export async function getStaticProps() {
  // Only what the cards and rows show; props must be JSON (no `undefined`).
  const posts = (await getAllFilesFrontMatter('blog')).map((post) => ({
    slug: post.slug,
    title: post.title || post.slug,
    subtitle: post.subtitle ?? null,
    status: post.status ?? null,
    summary: post.summary ?? null,
    date: post.date,
    tags: post.tags,
    source: post.notion ? 'notion' : 'mdx',
  }))

  return { props: { posts } }
}

export default function Home({ posts }) {
  return (
    <>
      <PageSEO title={siteMetadata.title} description={siteMetadata.description} />
      <p className="mb-10 border-b border-gray-200 pt-6 pb-8 text-lg leading-7 text-gray-500 dark:border-gray-700 dark:text-gray-400">
        {siteMetadata.description}
      </p>
      {posts.length === 0 ? 'No posts found.' : <PostBrowser posts={posts} />}
      {siteMetadata.newsletter.provider !== '' && (
        <div className="flex items-center justify-center pt-4">
          <NewsletterForm />
        </div>
      )}
    </>
  )
}
