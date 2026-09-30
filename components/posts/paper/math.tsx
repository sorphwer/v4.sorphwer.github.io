/**
 * KaTeX helpers for the paper page. Rendering happens through
 * `renderToString`, so the same components work during SSR and in the
 * client bundle; results are memoized because the paper repeats many
 * small inline formulas.
 */

import katex from 'katex'

const cache = new Map<string, string>()

function render(tex: string, displayMode: boolean): string {
  const key = (displayMode ? 'D' : 'I') + tex
  const hit = cache.get(key)
  if (hit) return hit
  const html = katex.renderToString(tex, {
    displayMode,
    throwOnError: false,
    strict: 'ignore',
  })
  cache.set(key, html)
  return html
}

/** Inline math. Over-wide expressions get `\allowbreak` at the call site. */
export function M({ t }: { t: string }) {
  return (
    <span
      className="[&_.katex]:text-[1.02em]"
      dangerouslySetInnerHTML={{ __html: render(t, false) }}
    />
  )
}

/** Display (block) math. */
export function MD({ t }: { t: string }) {
  return (
    <div
      className="my-4 overflow-x-auto py-1 text-[15px]"
      dangerouslySetInnerHTML={{ __html: render(t, true) }}
    />
  )
}
