/* eslint-disable react/display-name */
import { useMemo } from 'react'
import { getMDXComponent } from 'mdx-bundler/client'
import Image from './Image'
import CustomLink from './Link'
import TOCInline from './TOCInline'
import Pre from './Pre'
import { BlogNewsletterForm } from './NewsletterForm'
import { NotionRenderer } from 'react-notion-x'
import { getPageTitle } from 'notion-utils'
import dynamic from 'next/dynamic'

// Interactive post bodies live in components/posts/* and are registered here
// (webpack side) rather than imported from the .mdx: mdx-bundler would inline
// them into the serialized page props and `next/dynamic` would not work.
// Paper is server-rendered (English default) so its prose is in the static HTML;
// the film and glance figures are client-only (animation/layout measurement).
const PaperArticle = dynamic(() => import('./posts/paper/PaperArticle'))
const Film = dynamic(() => import('./posts/glance/film/Film'), { ssr: false })
// One chunk for all glance figures; `<Glance.Sea />` etc. in the .mdx.
const GLANCE_FIGURES = [
  'Sea',
  'Hard',
  'Pipe',
  'Mask',
  'Ground',
  'Schema',
  'Graph',
  'Search',
  'Cache',
  'Nums',
  'Entry',
  'Doc',
]
const Glance = Object.fromEntries(
  GLANCE_FIGURES.map((name) => [
    name,
    dynamic(() => import('./posts/glance/figures').then((m) => m[name]), { ssr: false }),
  ])
)

const Code = dynamic(() => import('react-notion-x/build/third-party/code').then((m) => m.Code))
const Collection = dynamic(() =>
  import('react-notion-x/build/third-party/collection').then((m) => m.Collection)
)
const Equation = dynamic(() =>
  import('react-notion-x/build/third-party/equation').then((m) => m.Equation)
)
const Pdf = dynamic(() => import('react-notion-x/build/third-party/pdf').then((m) => m.Pdf), {
  ssr: false,
})
const Modal = dynamic(() => import('react-notion-x/build/third-party/modal').then((m) => m.Modal), {
  ssr: false,
})
// Client-only: mdx-mermaid's Mermaid component calls mermaid.render in an effect.
const Mermaid = dynamic(() => import('mdx-mermaid/lib/Mermaid').then((m) => m.Mermaid), {
  ssr: false,
})
export const MDXComponents = {
  Image,
  TOCInline,
  a: CustomLink,
  pre: Pre,
  BlogNewsletterForm: BlogNewsletterForm,
  mermaid: Mermaid,
  Mermaid,
  PaperArticle,
  Film,
  Glance,
  wrapper: ({ components, layout, ...rest }) => {
    const Layout = require(`../layouts/${layout}`).default
    return <Layout {...rest} />
  },
}

export const MDXLayoutRenderer = ({ layout, mdxSource, recordMap, ...rest }) => {
  const MDXLayout = useMemo(() => getMDXComponent(mdxSource), [mdxSource])
  // const { recordMap,...reset } = rest
  const NotionJsx = recordMap ? (
    <NotionRenderer
      recordMap={recordMap}
      fullPage={true}
      darkMode={true}
      components={{
        Code,
        Collection,
        Equation,
        Modal,
        Pdf,
      }}
    />
  ) : (
    <span className="noNotion"></span>
  )
  const NotionTitle = recordMap ? getPageTitle(recordMap) : null

  return (
    <>
      <MDXLayout
        layout={layout}
        components={MDXComponents}
        NotionJsx={NotionJsx}
        NotionTitle={NotionTitle}
        {...rest}
      />
      {/* {recordMap && (
      <NotionRenderer recordMap={recordMap} fullPage={true} darkMode={true}/>
    )} */}
    </>
  )
}
