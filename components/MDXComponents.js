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
export const MDXComponents = {
  Image,
  TOCInline,
  a: CustomLink,
  pre: Pre,
  BlogNewsletterForm: BlogNewsletterForm,
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
