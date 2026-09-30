import { useState } from 'react'
import { TONE, Tag } from '../shared'

const ENTS = [
  {
    tone: 'ink',
    name: 'Zendesk 回票 agent',
    line: '回复客户之前，自动先查一遍知识库，把结论和原话一起放进回复草稿。',
    icon: (
      <>
        <path d="M6 4.5H18Q21 4.5 21 7.5V14Q21 17 18 17H11L6.5 20.5V17H6Q3 17 3 14V7.5Q3 4.5 6 4.5Z" />
        <circle cx="8" cy="10.8" r=".6" fill="currentColor" />
        <circle cx="12" cy="10.8" r=".6" fill="currentColor" />
        <circle cx="16" cy="10.8" r=".6" fill="currentColor" />
      </>
    ),
  },
  {
    tone: 'blue',
    name: 'Dify 外部知识库',
    line: '通过外部知识库 API 接入，任何 Dify 应用都能把它当知识库直接用。',
    icon: (
      <>
        <path d="M12 13.5L20.5 17.5L12 21.5L3.5 17.5Z" />
        <path d="M12 9L20.5 13L12 17L3.5 13Z" />
        <path d="M12 3.5L20.5 7.5L12 11.5L3.5 7.5Z" fill="currentColor" fillOpacity=".16" />
      </>
    ),
  },
  {
    tone: 'gray',
    name: '每天夜里自动同步',
    line: '当天关闭的工单，第二天就在图里。',
    icon: (
      <path d="M16.5 16.2A7.8 7.8 0 0 1 8.3 4.6A8 8 0 1 0 19.4 13.2A7.6 7.6 0 0 1 16.5 16.2Z" />
    ),
  },
]

export default function Entry() {
  const [sent, setSent] = useState(false)
  return (
    <figure className="fig wide" id="fig-entry">
      <div className="entries">
        <div className={sent ? 'panel reply sent' : 'panel reply'}>
          <div className="top">
            <span>回复草稿</span>
            <span className="sp">
              <Tag tone="blue" mono>
                #2948
              </Tag>
              <Tag tone="gray" mono>
                #3256
              </Tag>
            </span>
          </div>
          <div className="h">
            故障出在 sandbox 的 seccomp 初始化，
            <br />
            不在您的代码。
          </div>
          <div className="q">
            当前错误发生在 sandbox 初始化 DifySeccomp 阶段，不是代码逻辑本身的问题。
            <div className="by">— #2948 · comment_id=…5924</div>
          </div>
          <div className="act">
            <button className="btn primary" onClick={() => setSent((s) => !s)}>
              {sent ? '撤回' : '插入工单回复'}
            </button>
            <span className="done">
              <svg className="ico" viewBox="0 0 24 24" style={{ width: 18, height: 18 }}>
                <circle cx="12" cy="12" r="9" />
                <path d="M8 12.3l2.7 2.6L16 9.5" />
              </svg>
              已插入
            </span>
          </div>
        </div>
        <div>
          {ENTS.map((e) => (
            <div key={e.name} className="panel ent">
              <span className="ib" style={{ background: TONE[e.tone][1], color: TONE[e.tone][0] }}>
                <svg className="ico" viewBox="0 0 24 24">
                  {e.icon}
                </svg>
              </span>
              <div>
                <div className="nm">{e.name}</div>
                <div className="ln">{e.line}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <figcaption className="cap">
        左：开头那张新工单的回复草稿。结论旁边就是原话和出处，工程师确认后一键插入。
      </figcaption>
    </figure>
  )
}
