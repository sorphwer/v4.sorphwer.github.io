import { MDXLayoutRenderer } from '@/components/MDXComponents'
import { getFileBySlug } from '@/lib/mdx'
import { NotionAPI } from 'notion-client'
import { NotionRenderer } from 'react-notion-x'
const DEFAULT_LAYOUT = 'AuthorLayout'
// const DEFAULT_LAYOUT = 'NotionLayout'

//Next.js SSR
export async function getStaticProps() {
  const authorDetails = await getFileBySlug('authors', ['default'])
  //notion
  let recordMap = null
  if (authorDetails.frontMatter.notion) {
    const notion = new NotionAPI()
    recordMap = await notion.getPage(authorDetails.frontMatter.notion)
  } else {
    recordMap = null
  }
  return { props: { authorDetails, recordMap } }
}

export default function About({ authorDetails, recordMap }) {
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
