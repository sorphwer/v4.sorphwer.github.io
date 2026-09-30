import { Fragment, useState } from 'react'
import { T, useT } from '@/components/article/lang'
import { TONE } from '../shared'

const I = {
  doc: (
    <>
      <path d="M7 3.5H14L18 7.5V20.5H7Z" />
      <path d="M14 3.5V7.5H18M9.5 11H15.5M9.5 14H15.5M9.5 17H13" />
    </>
  ),
  mask: (
    <>
      <rect x="3.5" y="6" width="17" height="12" rx="2" />
      <path d="M7 12h3.5M13.5 12H17" strokeDasharray="1.6 1.8" />
    </>
  ),
  spark: (
    <>
      <path
        d="M10.5 5C11 9 12 10 16 10.5 12 11 11 12 10.5 16 10 12 9 11 5 10.5 9 10 10 9 10.5 5Z"
        fill="currentColor"
        fillOpacity=".15"
      />
      <path d="M18 3.5v3M16.5 5h3" />
    </>
  ),
  check: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M8.3 12.2L11 14.8 15.8 9.6" />
    </>
  ),
  shield: (
    <>
      <path d="M12 3L19.5 5.8V11.3C19.5 15.6 16.4 19.3 12 21C7.6 19.3 4.5 15.6 4.5 11.3V5.8Z" />
      <path d="M8.6 12.2L11 14.5 15.6 9.7" />
    </>
  ),
  vec: (
    <>
      <path d="M4 20L20 4M4 20l10-2M4 20l3-11" />
      <circle cx="20" cy="4" r="1.4" fill="currentColor" />
      <circle cx="14" cy="18" r="1.4" fill="currentColor" />
      <circle cx="7" cy="9" r="1.4" fill="currentColor" />
    </>
  ),
  graph: (
    <>
      <circle cx="5" cy="18" r="2.4" />
      <circle cx="12" cy="5.5" r="2.4" />
      <circle cx="19" cy="17" r="2.4" />
      <path d="M6.3 16l4.5-8.4M13.2 7.6l4.6 7.4M7.4 17.8h9.2" />
    </>
  ),
}

const GATE = [['Gate', '关'], 'pink']
const CACHE = [['Cache', '缓存'], 'blue']

const ST = [
  {
    nm: ['Raw ticket', '原始工单'],
    sm: ['Stored as-is', '原样存档'],
    ic: 'doc',
    tone: 'gray',
    txt: [
      'The ticket record and attachments, pulled from the ticketing system and left untouched. This is the only input to the whole system; everything else can be recomputed from it.',
      '从工单系统原样取回工单记录和附件，一字不改。它是整个系统唯一的输入，其余一切都能从它重新算出来。',
    ],
  },
  {
    nm: ['Render & mask', '渲染 · 脱敏'],
    sm: ['Gate 1', '第一道关'],
    ic: 'mask',
    tone: 'pink',
    badge: GATE,
    txt: [
      'Deterministically turns the ticket into text for the model, with every comment in time order and its id kept; names, emails and phone numbers become placeholders. This is the only version the model ever sees.',
      '确定性地把工单转成给模型读的文本，按时间排好每条回复并保留编号；同时把人名、邮箱、电话换成占位符。模型只见得到这一版。',
    ],
  },
  {
    nm: ['AI extraction', 'AI 提炼'],
    sm: ['Only LLM call', '唯一调用模型'],
    ic: 'spark',
    tone: 'blue',
    badge: CACHE,
    txt: [
      'Problem summary, solution summary (split into claims, each with a verbatim quote), keywords and external links. Cached by content: if neither the ticket nor the config changed, the model is not called again.',
      '问题摘要、解决方案摘要（拆成带原话的结论）、关键词、外部链接。按内容寻址缓存：工单和配置都没变，就不再调用模型。',
    ],
  },
  {
    nm: ['Verify sources', '校验出处'],
    sm: ['Pure function', '纯函数'],
    ic: 'check',
    tone: 'ink',
    txt: [
      'Checks sentence by sentence that each claim comes from the summary, that each quote appears verbatim in its comment, and that “resolved” is backed by evidence. A failure gets one targeted repair pass; if it still fails, it is marked unverified.',
      '逐句检查结论是否出自摘要、原话是否逐字存在于对应回复、“已解决”是否有证据。不过就定向修复一轮，仍不过则标为未核实。',
    ],
  },
  {
    nm: ['Pre-write check', '入库前再查'],
    sm: ['Gate 2', '第二道关'],
    ic: 'shield',
    tone: 'pink',
    badge: GATE,
    txt: [
      'Every field the model produced goes through the same masking rules again. Any hit blocks the whole ticket and nothing is written; the log records the rule and the field, never the value.',
      '模型输出的每一个字段再过一遍同一套脱敏规则。命中即整票拦下，一行不写；只记录规则和字段，不记录原值。',
    ],
  },
  {
    nm: ['Embedding', '向量化'],
    sm: ['Two summaries', '两段摘要'],
    ic: 'vec',
    tone: 'gray',
    badge: CACHE,
    txt: [
      'The problem summary and the solution summary each get one vector for semantic search. Cached by text: if a summary didn’t change, it isn’t re-embedded.',
      '问题摘要和解决方案摘要各生成一个向量，供语义检索使用。按文字内容缓存：摘要没变，就不重新向量化。',
    ],
  },
  {
    nm: ['Knowledge graph', '知识图谱'],
    sm: ['Nightly sync', '每晚同步'],
    ic: 'graph',
    tone: 'blue',
    txt: [
      'Newly closed tickets are added to the graph incrementally every night. When a prompt or rule changes, a new graph is rebuilt from scratch alongside, and only swapped in after passing the same set of test questions.',
      '每晚把新关闭的工单增量写入图；改了 prompt 或规则时，在旁边把新图完整重建，用同一组问题验收通过后再切换。',
    ],
  },
]

const LEGEND_TAG = { fontSize: 11, padding: '1px 6px' }

export default function Pipe() {
  const [sel, setSel] = useState(2)
  const t = useT()

  return (
    <figure className="fig wide" id="fig-pipe">
      <div className="panel wash">
        <div className="pipe">
          {ST.map((s, i) => (
            <Fragment key={s.ic}>
              {i ? <span className="ar" style={{ '--d': `${(i * 0.34).toFixed(2)}s` }} /> : null}
              <button className={`st${i === sel ? ' on' : ''}`} onClick={() => setSel(i)}>
                {s.badge ? (
                  <span className={`badge tag ${s.badge[1]}`}>{t(...s.badge[0])}</span>
                ) : null}
                <span
                  className="ic"
                  style={{ background: TONE[s.tone][1], color: TONE[s.tone][0] }}
                >
                  <svg className="ico" viewBox="0 0 24 24">
                    {I[s.ic]}
                  </svg>
                </span>
                <div className="nm">{t(...s.nm)}</div>
                <div className="sm">{t(...s.sm)}</div>
              </button>
            </Fragment>
          ))}
        </div>
        <div className="pipe-detail">
          <div className="k">
            {sel + 1} · {t(...ST[sel].nm)}
          </div>
          <div className="v">{t(...ST[sel].txt)}</div>
        </div>
        <div className="pipe-legend">
          <span>
            <span className="tag pink" style={LEGEND_TAG}>
              {t(...GATE[0])}
            </span>
            {t('Two masking gates', '两道脱敏关')}
          </span>
          <span>
            <span className="tag blue" style={LEGEND_TAG}>
              {t(...CACHE[0])}
            </span>
            {t('Not recomputed unless the content changed', '内容没变就不重算')}
          </span>
          <span>{t('Click any step for details', '点击任一步查看细节')}</span>
        </div>
      </div>
      <figcaption className="cap">
        <T
          en="The seven steps a ticket goes through, from raw text to the graph. Only “AI extraction” calls an LLM; verification, the masking checks and graph writes are all deterministic code."
          zh="一张工单从原文到图谱经过的七步。只有“AI 提炼”这一步调用大模型；校验、脱敏检查和入图都是确定性的代码。"
        />
      </figcaption>
    </figure>
  )
}
