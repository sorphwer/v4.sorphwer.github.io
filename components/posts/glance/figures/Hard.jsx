import { useEffect, useRef, useState } from 'react'
import { T, useT } from '@/components/article/lang'
import { TONE, Tag, useReveal } from '../shared'

const COLS = [
  {
    title: ['Hard to answer', '问题难回答'],
    tone: 'pink',
    icon: (
      <>
        <path d="M12 3.5C17 3.5 20.5 6.8 20.5 11S17 18.5 12 18.5C11 18.5 10 18.4 9.1 18.1L4.5 20.5L5.5 16.3C4.3 14.9 3.5 13 3.5 11C3.5 6.8 7 3.5 12 3.5Z" />
        <path d="M9.6 9.1C9.8 7.8 10.8 7 12.1 7C13.5 7 14.5 7.9 14.5 9.1C14.5 10.8 12.2 10.9 12.2 12.6" />
        <circle cx="12.2" cy="15.2" r=".8" fill="currentColor" />
      </>
    ),
  },
  {
    title: ['Hard to trust', '回答难被相信'],
    tone: 'pink',
    icon: (
      <>
        <path d="M12 3L19.5 5.8V11.3C19.5 15.6 16.4 19.3 12 21C7.6 19.3 4.5 15.6 4.5 11.3V5.8Z" />
        <path d="M12 8V12.6" />
        <circle cx="12" cy="15.6" r=".8" fill="currentColor" />
      </>
    ),
  },
]

const arrow = <span>→</span>

// [column, [en, zh] problem, [en, zh] fix, demo]
const ROWS = [
  [
    0,
    ['Mixed versions & error codes', '版本号、报错码混在一起'],
    ['Paste as-is and search', '原样贴进去，直接搜'],
    <>
      <Tag tone="gray" mono>
        #3412
      </Tag>
      <Tag tone="blue" mono>
        3.8.0
      </Tag>
      <Tag tone="ink" mono>
        panic: …
      </Tag>
    </>,
  ],
  [
    0,
    ['Vague symptom descriptions', '症状描述很模糊'],
    ['“Offline” finds the error', '说“连不上”，也能找到报错'],
    <>
      <Tag tone="blue">
        <T en="offline" zh="连不上" />
      </Tag>
      {arrow}
      <Tag tone="green" mono>
        ECONNREFUSED
      </Tag>
    </>,
  ],
  [
    0,
    ['The answer sits behind a link', '答案藏在外部链接里'],
    ['Docs and issues found too', '文档和 issue 一起找到'],
    <>
      <Tag tone="purple" mono>
        docs/…
      </Tag>
      <Tag tone="purple" mono>
        issue #232
      </Tag>
    </>,
  ],
  [
    1,
    ['How was this summary made?', '总结是怎么写出来的？'],
    ['Summaries are traceable', '每条总结都说得清来历'],
    <>
      <Tag tone="gray">
        <T en="ticket" zh="哪张工单" />
      </Tag>
      {arrow}
      <Tag tone="gray">
        <T en="call" zh="哪次调用" />
      </Tag>
      {arrow}
      <Tag tone="gray">
        <T en="prompt" zh="哪版 prompt" />
      </Tag>
    </>,
  ],
  [
    1,
    ['Which sentence is the source?', '原文到底是哪一句？'],
    ['The quote sits by the claim', '结论旁边就是原话'],
    <>
      <Tag tone="blue">
        <T en="claim" zh="结论" />
      </Tag>
      <span>↔</span>
      <Tag tone="ink">
        <T en="quote" zh="原话" />
      </Tag>
    </>,
  ],
  [
    1,
    ['Could personal data leak out?', '个人信息会不会泄露？'],
    ['Two filters before storage', '入库前拦两遍'],
    <>
      <Tag tone="pink" mono>
        [NAME_1]
      </Tag>
      <Tag tone="pink" mono>
        [EMAIL_1]
      </Tag>
    </>,
  ],
]

const NONE = ROWS.map(() => false)
const setAt = (i, f) => (d) => d.map((x, k) => (k === i ? f(x) : x))

export default function Hard() {
  const t = useT()
  const [gridRef, revealed] = useReveal(0.5)
  const [done, setDone] = useState(NONE)
  const timers = useRef([])

  // autoplay: tick the items one by one once the grid scrolls into view
  useEffect(() => {
    if (!revealed) return
    const ts = timers.current
    ROWS.forEach((_, i) => ts.push(setTimeout(() => setDone(setAt(i, () => true)), 500 + i * 650)))
    return () => ts.forEach(clearTimeout)
  }, [revealed])

  const reset = () => {
    timers.current.forEach(clearTimeout)
    setDone(NONE)
  }

  return (
    <figure className="fig wide" id="fig-hard">
      <div className="panel">
        <div className="hard" ref={gridRef}>
          {COLS.map((c, ci) => {
            const rows = ROWS.map((r, i) => [r, i]).filter(([r]) => r[0] === ci)
            const all = rows.every(([, i]) => done[i])
            return (
              <div className={`col${all ? ' all' : ''}`} key={ci}>
                <h4>
                  <span
                    className="ib"
                    style={{ background: TONE[c.tone][1], color: TONE[c.tone][0] }}
                  >
                    <svg className="ico" viewBox="0 0 24 24">
                      {c.icon}
                    </svg>
                  </span>
                  {t(...c.title)}
                </h4>
                {rows.map(([r, i]) => {
                  const toggle = () => setDone(setAt(i, (x) => !x))
                  return (
                    <div
                      className={`item${done[i] ? ' done' : ''}`}
                      role="button"
                      tabIndex={0}
                      key={i}
                      onClick={toggle}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          toggle()
                        }
                      }}
                    >
                      <span className="box">
                        <svg viewBox="0 0 12 12">
                          <path d="M2.5 6.3L5 8.6 9.6 3.6" />
                        </svg>
                      </span>
                      <div className="t">
                        <div className="pb">{t(...r[1])}</div>
                        <div className="fx">{t(...r[2])}</div>
                      </div>
                      <div className="was">{t(...r[1])}</div>
                      <div className="demo">{r[3]}</div>
                    </div>
                  )
                })}
              </div>
            )
          })}
        </div>
        <div className="row" style={{ marginTop: 14, justifyContent: 'space-between' }}>
          <span className="lbl">
            <T en="Click any item to see how we handle it" zh="点击任意一条，看我们怎么处理它" />
          </span>
          <button className="btn" onClick={reset}>
            <T en="Replay" zh="重来" />
          </button>
        </div>
      </div>
      <figcaption className="cap">
        <T
          en="Six concrete difficulties and how we address each. The left column is about “can we find it”, the right about “can we trust it”."
          zh="六个具体的难点，以及对应的做法。左边一列关于“搜得到”，右边一列关于“信得过”。"
        />
      </figcaption>
    </figure>
  )
}
