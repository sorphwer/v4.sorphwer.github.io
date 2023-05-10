import { MDXLayoutRenderer } from '@/components/MDXComponents'
import { getFileBySlug } from '@/lib/mdx'
import { NotionAPI } from 'notion-client'
import { NotionRenderer } from 'react-notion-x'
const DEFAULT_LAYOUT = 'AuthorLayout'
// const DEFAULT_LAYOUT = 'NotionLayout'

//Next.js SSR
export async function getStaticProps() {
  const notion = new NotionAPI()
  const recordMap = await notion.getPage('009dcf743a8e40cca778c55251123a9b')
  const authorDetails = await getFileBySlug('authors', ['default'])
  return { props: { authorDetails, recordMap } }
}

export default function About({ authorDetails, recordMap }) {
  const { mdxSource, frontMatter } = authorDetails

  return (
    <NotionRenderer recordMap={recordMap} fullPage={true} darkMode={true} />
    // <MDXLayoutRenderer
    //   layout={frontMatter.layout || DEFAULT_LAYOUT}
    //   mdxSource={mdxSource}
    //   frontMatter={frontMatter}
    // />
  )
}
