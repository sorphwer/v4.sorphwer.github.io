import { useEffect, useRef, useState } from 'react'

// Markdown/MDX headings are h1–h3; Notion renders its three heading levels as
// h2–h4 tagged `.notion-h1`–`.notion-h3`.
const HEADING_SELECTOR = 'h1, h2, h3, .notion-h'
const NOTION_LEVEL = /\bnotion-h([123])\b/
const SCROLL_OFFSET = 88
const BAR_WIDTH = [20, 14, 9]

function collectHeadings(root) {
  const found = []
  for (const el of root.querySelectorAll(HEADING_SELECTOR)) {
    if (el.closest('nav, figure, [data-toc-skip]') || el.getClientRects().length === 0) continue
    const text = el.textContent.replace(/\s+/g, ' ').trim()
    if (!text) continue
    const notion = NOTION_LEVEL.exec(el.className)
    found.push({ el, text, level: notion ? Number(notion[1]) : Number(el.tagName[1]) })
  }
  const top = Math.min(...found.map((h) => h.level))
  return found.map((h) => ({ ...h, depth: Math.min(h.level - top, 2) }))
}

/** Anchor for the URL hash: the heading's own id, or Notion's anchor child. */
function anchorId(el) {
  return el.id || el.querySelector('.notion-header-anchor')?.id || ''
}

/**
 * Notion-style reading rail: fixed at the right edge of the viewport, one short
 * bar per heading (length by depth, the heading being read in blue). Hovering or
 * focusing the rail opens a card listing every heading.
 *
 * Headings are read from the rendered DOM under `rootRef`, so MDX, Notion and
 * component-built posts all work, and the list follows content that changes
 * after mount (language toggle, client-only figures).
 */
export default function PostToc({ rootRef }) {
  const [headings, setHeadings] = useState([])
  const [active, setActive] = useState(-1)
  const listRef = useRef(null)

  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    let signature = ''
    let timer = null
    const refresh = () => {
      timer = null
      const next = collectHeadings(root)
      const nextSignature = next.map((h) => `${h.depth}:${h.text}`).join('\n')
      if (nextSignature === signature) return
      signature = nextSignature
      setHeadings(next)
    }
    refresh()
    // Throttled rather than debounced: animated figures mutate continuously.
    const observer = new MutationObserver(() => {
      if (timer === null) timer = setTimeout(refresh, 250)
    })
    observer.observe(root, { childList: true, subtree: true, characterData: true })
    return () => {
      observer.disconnect()
      clearTimeout(timer)
    }
  }, [rootRef])

  useEffect(() => {
    if (headings.length === 0) return
    let frame = 0
    const update = () => {
      frame = 0
      const line = window.innerHeight * 0.3
      let current = -1
      headings.forEach((h, i) => {
        if (h.el.getBoundingClientRect().top <= line) current = i
      })
      const atBottom =
        window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2
      setActive(atBottom && current >= 0 ? headings.length - 1 : current)
    }
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(update)
    }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [headings])

  // Keep the active entry inside the card's own scroll box without moving the page.
  useEffect(() => {
    const list = listRef.current
    const item = list?.children[active]
    if (!item) return
    if (item.offsetTop < list.scrollTop) list.scrollTop = item.offsetTop - 8
    else if (item.offsetTop + item.offsetHeight > list.scrollTop + list.clientHeight)
      list.scrollTop = item.offsetTop + item.offsetHeight - list.clientHeight + 8
  }, [active])

  if (headings.length < 2) return null

  const go = (index) => {
    const { el } = headings[index]
    const top = el.getBoundingClientRect().top + window.scrollY - SCROLL_OFFSET
    window.scrollTo({ top, behavior: 'smooth' })
    const id = anchorId(el)
    if (id) window.history.replaceState(null, '', `#${id}`)
  }

  const rowHeight = Math.max(5, Math.min(12, Math.floor(420 / headings.length)))

  return (
    <nav
      aria-label="Table of contents"
      className="group fixed right-2 top-1/2 z-40 hidden -translate-y-1/2 lg:block xl:right-5"
    >
      <ul aria-hidden="true" className="flex cursor-pointer flex-col items-end py-3 pl-6 pr-2">
        {headings.map((h, i) => (
          // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-noninteractive-element-interactions
          <li
            key={`${i}-${h.text}`}
            className="flex items-center"
            style={{ height: rowHeight }}
            onClick={() => go(i)}
          >
            <span
              className={`block h-0.5 rounded-full transition-colors duration-200 ${
                i === active
                  ? 'bg-primary-500 dark:bg-primary-400'
                  : 'bg-gray-300 group-hover:bg-gray-400 dark:bg-gray-600 dark:group-hover:bg-gray-500'
              }`}
              style={{ width: BAR_WIDTH[h.depth] }}
            />
          </li>
        ))}
      </ul>
      {/* Transparent padding bridges the rail and the card so the hover survives the gap. */}
      <div className="pointer-events-none absolute right-full top-1/2 -translate-y-1/2 translate-x-2 pr-1 opacity-0 transition duration-200 delay-150 group-focus-within:pointer-events-auto group-focus-within:translate-x-0 group-focus-within:opacity-100 group-focus-within:delay-0 group-hover:pointer-events-auto group-hover:translate-x-0 group-hover:opacity-100 group-hover:delay-0">
        <ol
          ref={listRef}
          className="max-h-[70vh] w-72 overflow-y-auto overscroll-contain rounded-xl border border-gray-200 bg-white/95 p-2 text-sm shadow-xl backdrop-blur dark:border-gray-700 dark:bg-gray-900/95"
        >
          {headings.map((h, i) => (
            <li key={`${i}-${h.text}`}>
              <a
                href={`#${anchorId(h.el)}`}
                onClick={(e) => {
                  e.preventDefault()
                  go(i)
                }}
                aria-current={i === active ? 'location' : undefined}
                className={`line-clamp-2 rounded-md py-1.5 pr-2 leading-5 transition-colors ${
                  i === active
                    ? 'text-primary-500 dark:text-primary-400'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-100'
                }`}
                style={{ paddingLeft: 8 + h.depth * 14 }}
                title={h.text}
              >
                {h.text}
              </a>
            </li>
          ))}
        </ol>
      </div>
    </nav>
  )
}
