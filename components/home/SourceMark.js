import Image from 'next/image'

/** Notion / MDX logo plus label, marking where a post's body comes from. */
export default function SourceMark({ source, className = '' }) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      {source === 'notion' ? (
        <Image src="/static/images/notion.svg" width={12} height={12} alt="" />
      ) : (
        <span className="inline-flex rounded-sm bg-white px-0.5">
          <Image src="/static/images/mdx.png" width={22} height={9} alt="" />
        </span>
      )}
      {source === 'notion' ? 'Notion' : 'MDX'}
    </span>
  )
}
