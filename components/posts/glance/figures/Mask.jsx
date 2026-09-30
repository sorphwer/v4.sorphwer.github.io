import { useState } from 'react'
import { T, useLang, useT } from '@/components/article/lang'
import { Tag } from '../shared'

const RULES = [
  { kind: 'EMAIL', re: /[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+/g },
  { kind: 'PHONE', re: /(?<!\d)1[3-9]\d[ -]?\d{4}[ -]?\d{4}(?!\d)/g },
  { kind: 'PHONE', re: /(?<![\w+])\+\d{1,3}(?:[ -]?\d{2,4}){2,4}(?!\d)/g },
  { kind: 'IP', re: /(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])/g },
]

const findPII = (text, names) => {
  const hits = []
  names.forEach((n) => {
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
const mask = (text, names) => {
  const hits = findPII(text, names)
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

// Per-language sample ticket, its name list, and the address gate 2 catches.
const SAMPLES = {
  en: {
    names: ['Amy Lin', 'Tom Reed'],
    text: 'From: Amy Lin <amy.lin@example.com>\nPhone: +1 415 555 0132\n\nAmy Lin: every code node fails on 3.9.5:\nprocess exited with code -1\npanic: could not create filter goroutine 17\nmain.DifySeccomp(…)\n\nThe sandbox is at 10.0.3.17, please help.\nReply to amy.lin@example.com and cc Tom Reed <tom.reed@example.org>.',
    leak: 'a.lin@corp-example.com',
  },
  zh: {
    names: ['王小明', '李雷'],
    text: '提交人：王小明 <xiaoming@example.com>\n电话：138 0013 8000\n\n王小明：升级到 3.9.5 之后，代码节点全部报错：\nprocess exited with code -1\npanic: could not create filter goroutine 17\nmain.DifySeccomp(…)\n\nsandbox 部署在 10.0.3.17，麻烦尽快看一下。\n结果请发 xiaoming@example.com，并抄送李雷 li.lei@example.org。',
    leak: 'wang.xm@corp-example.cn',
  },
}

// gate 2
const OUT_OK = {
  summary:
    'The failure occurred during Sandbox DifySeccomp initialization, not in the customer’s Python logic.',
  keywords: 'sandbox · seccomp · code node',
  links: 'github.com/langgenius/dify-sandbox/issues/232',
}
const outLeak = (email) => ({
  ...OUT_OK,
  summary: `${OUT_OK.summary} The attached screenshot shows the console logged in as ${email}.`,
})
const LBL = {
  summary: ['Summary', '摘要'],
  keywords: ['Keywords', '关键词'],
  links: ['Links', '链接'],
}

function Gate2({ sample }) {
  const t = useT()
  const [leak, setLeak] = useState(false)
  const o = leak ? outLeak(sample.leak) : OUT_OK
  let hitField = null
  let hitKind = null
  const rows = Object.keys(LBL).map((k) => {
    const hits = findPII(o[k], sample.names)
    if (hits.length && !hitField) {
      hitField = t(...LBL[k])
      hitKind = hits[0].kind
    }
    const nodes = splice(o[k], hits, (h, j) => (
      <span className="leak" key={j}>
        {h.v}
      </span>
    ))
    return [
      <span className="k" key={`${k}k`}>
        {t(...LBL[k])}
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
          <b style={{ fontSize: 15 }}>{t('Gate 2: before writing', '第二道关：入库前')}</b>
          <span className="lbl" style={{ marginLeft: 10 }}>
            {t(
              'Every piece of model output is checked again with the same rules',
              '模型的每一段输出，都用同一套规则再查一遍'
            )}
          </span>
        </div>
        <button className="btn" onClick={() => setLeak(!leak)}>
          {leak
            ? t('Restore normal output', '恢复正常输出')
            : t(
                'Simulate: an email from a screenshot lands in the summary',
                '模拟：截图里的邮箱被写进了摘要'
              )}
        </button>
      </div>
      <div className="fields">{rows}</div>
      {hitField ? (
        <div className="verdict bad">
          <b>{t('Whole ticket blocked', '整张工单拦下')}</b>
          <span>{t('Nothing is written', '不写入任何数据')}</span>
          <span className="sm">
            <T
              en={`Log records only: rule ${hitKind} · field “${hitField}”`}
              zh={`日志只记：规则 ${hitKind} · 字段“${hitField}”`}
            />
          </span>
        </div>
      ) : (
        <div className="verdict ok">
          <b>{t('Passed', '通过')}</b>
          <span>{t('No field matched any rule', '所有字段都没有命中规则')}</span>
          <span className="sm">{t('OK to write', '可以入库')}</span>
        </div>
      )}
    </div>
  )
}

export default function Mask() {
  const t = useT()
  const lang = useLang()
  const sample = SAMPLES[lang]
  // Edits belong to one language; switching language resets to that sample.
  const [edit, setEdit] = useState(null)
  const text = edit && edit.lang === lang ? edit.text : sample.text
  const r = mask(text, sample.names)

  return (
    <figure className="fig wide" id="fig-mask">
      <div className="panel">
        <div className="mask">
          <div>
            <div className="hd">
              <span>{t('Ticket (editable)', '工单原文（可编辑）')}</span>
              <span className="row">
                <span>{t('Name list', '本工单名单')}</span>
                {sample.names.map((n) => (
                  <Tag tone="gray" key={n}>
                    {n}
                  </Tag>
                ))}
              </span>
            </div>
            <textarea
              spellCheck={false}
              aria-label={t('Original ticket', '工单原文')}
              value={text}
              onChange={(e) => setEdit({ lang, text: e.target.value })}
            />
          </div>
          <div>
            <div className="hd">
              <span>{t('What the model sees', '模型看到的版本')}</span>
              <span>
                <T
                  en={`${r.n} replacements · ${r.uniq} distinct values`}
                  zh={`替换 ${r.n} 处 · ${r.uniq} 个不同的值`}
                />
              </span>
            </div>
            <div className="out" aria-live="polite">
              {r.nodes}
            </div>
          </div>
        </div>
        <Gate2 sample={sample} />
      </div>
      <figcaption className="cap">
        <T
          en="Sample text is fictional. The real rules are more numerous, but the structure is the same: shared rules plus each ticket’s own name list, and the same value always gets the same placeholder."
          zh="示例文字为虚构。真实规则比这里多，但结构一样：通用规则 + 每张工单自己的名单，同值同号。"
        />
      </figcaption>
    </figure>
  )
}
