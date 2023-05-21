import { MDXLayoutRenderer } from '@/components/MDXComponents'
import { getFileBySlug } from '@/lib/mdx'
import { NotionAPI } from 'notion-client'
import { NotionRenderer } from 'react-notion-x'
const DEFAULT_LAYOUT = 'AuthorLayout'
// const DEFAULT_LAYOUT = 'NotionLayout'

//Next.js SSR
export async function getStaticProps() {
  const aboutDetails = await getFileBySlug('about', ['default'])
  //notion
  let recordMap = null
  if (aboutDetails.frontMatter.notion) {
    const notion = new NotionAPI()
    recordMap = await notion.getPage(aboutDetails.frontMatter.notion)
  } else {
    recordMap = null
  }
  return { props: { aboutDetails, recordMap } }
}

export default function About({ aboutDetails, recordMap }) {
  const { mdxSource, frontMatter } = aboutDetails

  return (
    <MDXLayoutRenderer
      layout={frontMatter.layout || DEFAULT_LAYOUT}
      mdxSource={mdxSource}
      recordMap={recordMap}
      frontMatter={frontMatter}
    />
  )
}
