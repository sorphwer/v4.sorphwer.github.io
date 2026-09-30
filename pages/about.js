import { MDXLayoutRenderer } from '@/components/MDXComponents'
import { getFileBySlug } from '@/lib/mdx'
import { getNotionPage } from '@/lib/notion'
const DEFAULT_LAYOUT = 'AuthorLayout'

//Next.js SSR
export async function getStaticProps() {
  const aboutDetails = await getFileBySlug('about', ['default'])
  //notion
  let recordMap = null
  if (aboutDetails.frontMatter.notion) {
    recordMap = await getNotionPage(aboutDetails.frontMatter.notion)
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
