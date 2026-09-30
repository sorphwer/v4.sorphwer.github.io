import { useState } from 'react'
import { Tag } from '../shared'

const NAMES = ['王小明', '李雷']
const RULES = [
  { kind: 'EMAIL', re: /[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+/g },
  { kind: 'PHONE', re: /(?<!\d)1[3-9]\d[ -]?\d{4}[ -]?\d{4}(?!\d)/g },
  { kind: 'IP', re: /(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])/g },
]

const findPII = (text) => {
  const hits = []
  NAMES.forEach((n) => {
    let i = 0
    while ((i = text.indexOf(n, i)) !== -1) {
      hits.push({ s: i, e: i + n.length, kind: 'NAME', v: n })
      i += n.length
    }
  })
  RULES.forEach((r) => {
    for (const m of text.matchAll(r.re)) {
      hits.push({ s: m.index, e: m.index + m[0].length, kind: r.kind, v: m[0] })
    }
  })
  hits.sort((a, b) => a.s - b.s || b.e - a.e)
  const kept = []
  let end = -1
  hits.forEach((h) => {
    if (h.s >= end) {
      kept.push(h)
      end = h.e
    }
  })
  return kept
}

/** Splits `text` around its PII hits; `wrap(hit, k)` renders each hit. */
const splice = (text, hits, wrap) => {
  const out = []
  let pos = 0
  hits.forEach((h, k) => {
    out.push(text.slice(pos, h.s), wrap(h, k))
    pos = h.e
  })
  out.push(text.slice(pos))
  return out
}

// Same value → same placeholder within one ticket, numbered per kind.
const mask = (text) => {
  const hits = findPII(text)
  const ids = {}
  const cnt = {}
  const nodes = splice(text, hits, (h, k) => {
    const key = h.kind + '\u0000' + h.v
    if (!ids[key]) {
      cnt[h.kind] = (cnt[h.kind] || 0) + 1
      ids[key] = '[' + h.kind + '_' + cnt[h.kind] + ']'
    }
    return (
      <span className="tk" title={h.v} key={k}>
        {ids[key]}
      </span>
    )
  })
  return { nodes, n: hits.length, uniq: Object.keys(ids).length }
}

const SAMPLE =
  '提交人：王小明 <xiaoming@example.com>\n电话：138 0013 8000\n\n王小明：升级到 3.9.5 之后，代码节点全部报错：\nprocess exited with code -1\npanic: could not create filter goroutine 17\nmain.DifySeccomp(…)\n\nsandbox 部署在 10.0.3.17，麻烦尽快看一下。\n结果请发 xiaoming@example.com，并抄送李雷 li.lei@example.org。'

// gate 2
const OUT_OK = {
  summary:
    'The failure occurred during Sandbox DifySeccomp initialization, not in the customer’s Python logic.',
  keywords: 'sandbox · seccomp · code node',
  links: 'github.com/langgenius/dify-sandbox/issues/232',
}
const OUT_LEAK = {
  ...OUT_OK,
  summary:
    'The failure occurred during Sandbox DifySeccomp initialization, not in the customer’s Python logic. The attached screenshot shows the console logged in as wang.xm@corp-example.cn.',
}
const LBL = { summary: '摘要', keywords: '关键词', links: '链接' }

function Gate2() {
  const [leak, setLeak] = useState(false)
  const o = leak ? OUT_LEAK : OUT_OK
  let hitField = null
  let hitKind = null
  const rows = Object.keys(LBL).map((k) => {
    const hits = findPII(o[k])
    if (hits.length && !hitField) {
      hitField = LBL[k]
      hitKind = hits[0].kind
    }
    const nodes = splice(o[k], hits, (h, j) => (
      <span className="leak" key={j}>
        {h.v}
      </span>
    ))
    return [
      <span className="k" key={`${k}k`}>
        {LBL[k]}
      </span>,
      <span className="v" key={`${k}v`}>
        {nodes}
      </span>,
    ]
  })

  return (
    <div className="gate2">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div>
          <b style={{ fontSize: 15 }}>第二道关：入库前</b>
          <span className="lbl" style={{ marginLeft: 10 }}>
            模型的每一段输出，都用同一套规则再查一遍
          </span>
        </div>
        <button className="btn" onClick={() => setLeak(!leak)}>
          {leak ? '恢复正常输出' : '模拟：截图里的邮箱被写进了摘要'}
        </button>
      </div>
      <div className="fields">{rows}</div>
      {hitField ? (
        <div className="verdict bad">
          <b>整张工单拦下</b>
          <span>不写入任何数据</span>
          <span className="sm">
            日志只记：规则 {hitKind} · 字段“{hitField}”
          </span>
        </div>
      ) : (
        <div className="verdict ok">
          <b>通过</b>
          <span>所有字段都没有命中规则</span>
          <span className="sm">可以入库</span>
        </div>
      )}
    </div>
  )
}

export default function Mask() {
  const [text, setText] = useState(SAMPLE)
  const r = mask(text)

  return (
    <figure className="fig wide" id="fig-mask">
      <div className="panel">
        <div className="mask">
          <div>
            <div className="hd">
              <span>工单原文（可编辑）</span>
              <span className="row">
                <span>本工单名单</span>
                {NAMES.map((n) => (
                  <Tag tone="gray" key={n}>
                    {n}
                  </Tag>
                ))}
              </span>
            </div>
            <textarea
              spellCheck={false}
              aria-label="工单原文"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
          <div>
            <div className="hd">
              <span>模型看到的版本</span>
              <span>
                替换 {r.n} 处 · {r.uniq} 个不同的值
              </span>
            </div>
            <div className="out" aria-live="polite">
              {r.nodes}
            </div>
          </div>
        </div>
        <Gate2 />
      </div>
      <figcaption className="cap">
        示例文字为虚构。真实规则比这里多，但结构一样：通用规则 + 每张工单自己的名单，同值同号。
      </figcaption>
    </figure>
  )
}
