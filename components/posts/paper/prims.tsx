/**
 * Shared typographic primitives for the paper page. Every section file
 * composes these so the whole paper keeps one visual voice (Geist tokens,
 * identical table/callout/figure treatments). Headings, paragraphs and
 * their rhythm follow the site's `prose` scale (tailwind.config.js
 * `typography`): 24px bold h2, 20px h3, 16px/28px body with 20px gaps.
 */

import type { ReactNode } from 'react'
import { T } from '@/components/article/lang'

/** Top-level numbered section. `num` is the printed section number ("04"). */
export function Section({
  id,
  num,
  en,
  zh,
  children,
}: {
  id: string
  num: string
  en: string
  zh: string
  children: ReactNode
}) {
  return (
    <section id={id} className="mt-14 border-t border-ds-gray-alpha-400 pt-8 scroll-mt-24">
      <h2 className="flex items-baseline gap-3.5 text-2xl font-bold leading-8 tracking-tight text-ds-gray-1000">
        <span className="font-geist-mono text-[13px] font-normal text-ds-gray-700">{num}</span>{' '}
        <span className="text-balance">
          <T en={en} zh={zh} />
        </span>
      </h2>
      {children}
    </section>
  )
}

export function H3({ id, en, zh }: { id?: string; en: string; zh: string }) {
  return (
    <h3 id={id} className="mt-8 text-xl font-semibold leading-8 text-ds-gray-1000 scroll-mt-24">
      <T en={en} zh={zh} />
    </h3>
  )
}

export function H4({ en, zh }: { en: string; zh: string }) {
  return (
    <h4 className="mt-6 text-base font-semibold leading-6 text-ds-gray-1000">
      <T en={en} zh={zh} />
    </h4>
  )
}

/** Body paragraph. Pass `en`/`zh` as rich fragments. */
export function P({ en, zh }: { en: ReactNode; zh: ReactNode }) {
  return (
    <p className="mt-5 text-base leading-7 text-pretty">
      <T en={en} zh={zh} />
    </p>
  )
}

/** Bold lead-in for a paragraph ("Problem domain." / "问题域。"). */
export function Lead({ children }: { children: ReactNode }) {
  return <strong className="font-semibold text-ds-gray-1000">{children}</strong>
}

/** Inline code chip. */
export function C({ children }: { children: ReactNode }) {
  return (
    <code className="rounded bg-ds-gray-alpha-100 px-[5px] py-px font-geist-mono text-[12.5px]">
      {children}
    </code>
  )
}

/** Inline citation marker: <Cite n={4} /> renders [4]. */
export function Cite({ n }: { n: number | string }) {
  return <span className="text-[13px] text-ds-blue-900">[{n}]</span>
}

const CHANNEL_STYLE: Record<string, string> = {
  gr: 'bg-ds-blue-100 text-ds-blue-900 shadow-[inset_0_0_0_1px_var(--ds-blue-400)]',
}

/** Channel pill: iv / sv / ft / kw are neutral, gr is the accent channel. */
export function Chan({ c }: { c: 'iv' | 'sv' | 'ft' | 'kw' | 'gr' }) {
  return (
    <span
      className={`inline-block rounded-full px-[7px] font-geist-mono text-[11px] font-medium leading-4 align-[1px] ${
        CHANNEL_STYLE[c] ??
        'bg-ds-gray-100 text-ds-gray-900 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]'
      }`}
    >
      {c}
    </span>
  )
}

/** Solid mono tag chip (I1/I2/I3, P7, C1..C7, pipeline stage labels). */
export function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="inline-block rounded bg-ds-gray-1000 px-1.5 font-geist-mono text-[11px] font-medium leading-4 text-ds-background-100">
      {children}
    </span>
  )
}

/** Invariant / proposition callout card (I1, I2, I3, P7). */
export function Invariant({ tag, en, zh }: { tag: ReactNode; en: ReactNode; zh: ReactNode }) {
  return (
    <div className="rounded-md bg-ds-background-200 p-[14px_18px] text-sm leading-[1.85] shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
      <span className="mr-2 inline-block align-[1px]">
        <Tag>{tag}</Tag>
      </span>
      <T en={en} zh={zh} />
    </div>
  )
}

/** Numbered contribution row (C1..C7 in the introduction). */
export function Contribution({ tag, en, zh }: { tag: string; en: ReactNode; zh: ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <span className="mt-[5px] flex-none">
        <Tag>{tag}</Tag>
      </span>
      <span className="text-base leading-7">
        <T en={en} zh={zh} />
      </span>
    </div>
  )
}

/**
 * Figure wrapper: content block + numbered caption. `num` prints as
 * "Figure 4 ·" / "图 4 ·".
 */
export function Fig({
  num,
  en,
  zh,
  children,
}: {
  num: number
  en: ReactNode
  zh: ReactNode
  children: ReactNode
}) {
  return (
    <figure className="mt-7">
      {children}
      <figcaption className="mt-3 text-[12.5px] leading-[1.75] text-ds-gray-900">
        <span className="font-semibold text-ds-gray-1000">
          <T en={`Figure ${num}`} zh={`图 ${num}`} />
        </span>{' '}
        · <T en={en} zh={zh} />
      </figcaption>
    </figure>
  )
}

/** Table caption line: "Table 2 · ..." / "表 2 · ...". */
export function TableCaption({ num, en, zh }: { num: number; en: ReactNode; zh: ReactNode }) {
  return (
    <div className="mb-2 text-[12.5px] leading-[1.7] text-ds-gray-900">
      <span className="font-semibold text-ds-gray-1000">
        <T en={`Table ${num}`} zh={`表 ${num}`} />
      </span>{' '}
      · <T en={en} zh={zh} />
    </div>
  )
}

/** Scrollable ring-inset table shell. */
export function TableShell({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-md shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
      <table className="w-full border-collapse text-[13px] leading-relaxed">{children}</table>
    </div>
  )
}

const ALIGN_CLASS: Record<'left' | 'right' | 'center', string> = {
  left: 'text-left',
  right: 'text-right',
  center: 'text-center',
}

export function Th({
  align = 'left',
  children,
}: {
  align?: 'left' | 'right' | 'center'
  children?: ReactNode
}) {
  return (
    <th
      className={`border-b border-ds-gray-alpha-400 bg-ds-background-200 px-3.5 py-[9px] text-xs font-medium text-ds-gray-900 ${ALIGN_CLASS[align]}`}
    >
      {children}
    </th>
  )
}

export function Td({
  align = 'left',
  mono = false,
  strong = false,
  muted = false,
  last = false,
  children,
}: {
  align?: 'left' | 'right' | 'center'
  mono?: boolean
  strong?: boolean
  muted?: boolean
  last?: boolean
  children?: ReactNode
}) {
  return (
    <td
      className={[
        'px-3.5 py-2',
        last ? '' : 'border-b border-ds-gray-alpha-100',
        ALIGN_CLASS[align],
        mono ? 'font-geist-mono text-[12.5px]' : '',
        strong ? 'font-semibold text-ds-blue-900' : '',
        muted ? 'text-ds-gray-700' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {children}
    </td>
  )
}

/** Panel used for diagram figures (star schema, pipeline, mining flow). */
export function DiagramPanel({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-md bg-ds-background-200 p-6 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
      {children}
    </div>
  )
}

/** A single node row inside a diagram (pipeline stages, flow steps). */
export function DiagramNode({
  label,
  children,
  emphasis = false,
}: {
  label?: ReactNode
  children: ReactNode
  emphasis?: boolean
}) {
  return (
    <div
      className={`flex items-center gap-3 rounded-md px-4 py-2.5 text-[13px] ${
        emphasis
          ? 'bg-ds-gray-1000 text-ds-background-100'
          : 'bg-ds-background-100 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]'
      }`}
    >
      {label != null && (
        <span className="flex-none font-geist-mono text-[11px] text-ds-gray-700">{label}</span>
      )}
      <span className="min-w-0">{children}</span>
    </div>
  )
}

/** Downward connector between diagram nodes, optional annotation. */
export function DiagramArrow({ note }: { note?: ReactNode }) {
  return (
    <div className="self-center py-0.5 text-center text-sm leading-snug text-ds-gray-600">
      ↓
      {note != null && (
        <span className="ml-1.5 font-geist-mono text-[10.5px] text-ds-gray-800">{note}</span>
      )}
    </div>
  )
}
