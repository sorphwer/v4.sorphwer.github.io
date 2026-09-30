import { MDXLayoutRenderer } from '@/components/MDXComponents'
import { getFileBySlug } from '@/lib/mdx'
import { getNotionPage } from '@/lib/notion'
const DEFAULT_LAYOUT = 'AuthorLayout'

//Next.js SSR
export async function getStaticProps() {
  const authorDetails = await getFileBySlug('authors', ['default'])
  //notion
  let recordMap = null
  if (authorDetails.frontMatter.notion) {
    recordMap = await getNotionPage(authorDetails.frontMatter.notion)
  }
  return { props: { authorDetails, recordMap } }
}

export default function Profile({ authorDetails, recordMap }) {
  const { mdxSource, frontMatter } = authorDetails

  return (
    <MDXLayoutRenderer
      layout={frontMatter.layout || DEFAULT_LAYOUT}
      mdxSource={mdxSource}
      recordMap={recordMap}
      frontMatter={frontMatter}
    />
  )
}
