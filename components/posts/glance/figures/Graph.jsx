import { useState } from 'react'
import { T, useT } from '@/components/article/lang'
import { G, KIND, TONE, Tag, featR } from '../shared'

const RARITIES = G.feats.map((x) => x.rarity)
const MAX_R = Math.max(...RARITIES)
const MIN_R = Math.min(...RARITIES)

const featTone = (f) => TONE[KIND[G.feats[f].kind].tone]

const FeatTag = ({ f }) => (
  <Tag tone={KIND[G.feats[f].kind].tone} mono>
    {G.feats[f].key}
  </Tag>
)

/** Tickets sharing a feature with ticket `i`, scored by the summed rarity of what they share. */
const neighbours = (i) => {
  const me = G.tickets[i]
  const score = new Map()
  me.fs.forEach((f) =>
    G.byFeat[f].forEach((j) => {
      if (j === i) return
      const cur = score.get(j) || { j, s: 0, fs: [] }
      cur.s += G.feats[f].rarity
      cur.fs.push(f)
      score.set(j, cur)
    })
  )
  return Array.from(score.values()).sort((a, b) => b.s - a.s || a.j - b.j)
}

/** Which nodes/edges are lit (`on`) and what the side panel says for a selection. */
const focusOf = (sel) => {
  if (sel.t != null) {
    const i = sel.t
    const me = G.tickets[i]
    const nb = neighbours(i)
    const nbSet = new Set(nb.map((n) => n.j))
    return {
      tickets: new Set([i, ...nbSet]),
      feats: new Set(me.fs),
      edge: (e) => e.t === i || (me.fs.includes(e.f) && nbSet.has(e.t)),
      ticket: { i, me, nb },
    }
  }
  const j = sel.f
  return {
    tickets: new Set(G.byFeat[j]),
    feats: new Set([j]),
    edge: (e) => e.f === j,
    feat: j,
  }
}

function TicketSide({ me, nb }) {
  const t = useT()
  const max = nb.length ? nb[0].s : 1
  return (
    <>
      <div className="mut">{t('Ticket', '工单')}</div>
      <div className="ttl">
        <span style={{ fontFamily: 'var(--mono)' }}>#{me.id}</span>
      </div>
      {me.title && (
        <div style={{ color: 'var(--mut)', fontSize: '13.5px', marginBottom: 6 }}>
          {t(...me.title)}
        </div>
      )}
      <div className="row" style={{ margin: '6px 0 12px' }}>
        {me.fs.map((f) => (
          <FeatTag key={f} f={f} />
        ))}
      </div>
      <div className="mut" style={{ marginBottom: 4 }}>
        {t(
          `Linked to ${nb.length} tickets through these ${me.fs.length} features. Scored by the rarity of what they share, the nearest neighbours:`,
          `通过这 ${me.fs.length} 个特征连到 ${nb.length} 张工单。按共享特征的稀有度打分，最近的邻居：`
        )}
      </div>
      {nb.slice(0, 6).map((n) => (
        <div className="nb" key={n.j}>
          <span className="id">#{G.tickets[n.j].id}</span>
          <span className="sh">
            {n.fs.map((f) => (
              <FeatTag key={f} f={f} />
            ))}
          </span>
          <span className="meter">
            <i style={{ width: `${(n.s / max) * 100}%` }} />
          </span>
        </div>
      ))}
    </>
  )
}

function FeatSide({ j }) {
  const t = useT()
  const f = G.feats[j]
  const rr = (f.rarity - MIN_R) / (MAX_R - MIN_R || 1)
  return (
    <>
      <div className="mut">{t(...KIND[f.kind].name)}</div>
      <div className="ttl">
        <FeatTag f={j} />
      </div>
      <div style={{ margin: '10px 0 4px' }}>
        <T
          en={
            <>
              Shared by <b>{f.deg}</b> tickets
            </>
          }
          zh={
            <>
              被 <b>{f.deg}</b> 张工单共享
            </>
          }
        />
      </div>
      <div className="mut">{t('Rarity', '稀有度')}</div>
      <div className="meter" style={{ height: 8, margin: '4px 0 12px' }}>
        <i style={{ width: `${8 + rr * 92}%` }} />
      </div>
      <div className="mut">
        {t(
          'The fewer tickets share it, the stronger the evidence that two tickets are related. Rarity here is 1 / log₂(1 + tickets), for illustration only.',
          '共享它的工单越少，“这两张工单相关”的证据就越强。这里的稀有度取 1 / log₂(1 + 工单数)，仅作示意。'
        )}
      </div>
    </>
  )
}

export default function Graph() {
  const t = useT()
  // `pinned` is the clicked selection; `hover` overrides it until the pointer leaves the svg.
  const [pinned, setPinned] = useState({ t: 0 })
  const [hover, setHover] = useState(null)
  const focus = focusOf(hover || pinned)

  const pin = (sel) => {
    setPinned(sel)
    setHover(null)
  }
  const nodeProps = (sel) => ({
    tabIndex: 0,
    onMouseOver: () => setHover(sel),
    onClick: (e) => {
      e.stopPropagation()
      pin(sel)
    },
    onKeyDown: (e) => {
      if (e.key === 'Enter') pin(sel)
    },
  })
  const on = (base, lit) => (lit ? `${base} on` : base)

  return (
    <figure className="fig wide" id="fig-graph">
      <div className="panel">
        <div className="gx">
          <div>
            <svg
              className="focus"
              viewBox="0 0 1000 560"
              role="img"
              aria-label={t('Interactive knowledge graph (illustrative)', '可交互的知识图谱示意')}
              onMouseLeave={() => setHover(null)}
              onClick={() => pin({ t: 0 })}
            >
              <g>
                {G.edges.map((e, k) => {
                  const t = G.tickets[e.t]
                  const f = G.feats[e.f]
                  const lit = focus.edge(e)
                  return (
                    <line
                      key={k}
                      className={on('e', lit)}
                      x1={t.x}
                      y1={t.y}
                      x2={f.x}
                      y2={f.y}
                      style={lit ? { stroke: featTone(e.f)[0] } : undefined}
                    />
                  )
                })}
              </g>
              <g>
                {G.feats.map((f, j) => {
                  const [fg, bg] = featTone(j)
                  return (
                    <g key={j} className={on('fn', focus.feats.has(j))} {...nodeProps({ f: j })}>
                      <circle cx={f.x} cy={f.y} r={featR(f) + 5} style={{ fill: bg }} />
                      <circle
                        cx={f.x}
                        cy={f.y}
                        r={featR(f)}
                        strokeWidth={1.6}
                        style={{ fill: fg, stroke: 'var(--card)' }}
                      />
                    </g>
                  )
                })}
              </g>
              <g>
                {G.tickets.map((tk, i) => (
                  <circle
                    key={i}
                    className={on(i === 0 ? 'tn me' : 'tn', focus.tickets.has(i))}
                    cx={tk.x}
                    cy={tk.y}
                    r={i === 0 ? 6.5 : 4.8}
                    {...nodeProps({ t: i })}
                  />
                ))}
              </g>
              <g>
                {G.feats.map((f, j) => (
                  <text
                    key={j}
                    className={on('fl', focus.feats.has(j))}
                    x={f.x + featR(f) + 7}
                    y={f.y + 4.5}
                    style={{ fill: featTone(j)[0] }}
                  >
                    {f.key}
                  </text>
                ))}
                {focus.ticket && (
                  <text
                    className="fl on"
                    x={focus.ticket.me.x - 11}
                    y={focus.ticket.me.y + 4.5}
                    textAnchor="end"
                    style={{ fill: 'var(--ink)', fontFamily: 'var(--mono)', fontWeight: 600 }}
                  >
                    #{focus.ticket.me.id}
                  </text>
                )}
              </g>
            </svg>
            <div className="legend">
              <span>
                <i style={{ background: 'var(--ink)', opacity: 0.8 }} />
                {t('Ticket', '工单')}
              </span>
              <span>
                <i style={{ background: 'var(--green)' }} />
                {t(...KIND.kw.name)}
              </span>
              <span>
                <i style={{ background: 'var(--blue)' }} />
                {t(...KIND.ver.name)}
              </span>
              <span>
                <i style={{ background: 'var(--purple)' }} />
                {t(...KIND.link.name)}
              </span>
              <span>
                {t('Bigger feature node = more tickets share it', '特征节点越大，共享它的工单越多')}
              </span>
            </div>
          </div>
          <div className="side">
            {focus.ticket ? (
              <TicketSide me={focus.ticket.me} nb={focus.ticket.nb} />
            ) : (
              <FeatSide j={focus.feat} />
            )}
          </div>
        </div>
      </div>
      <figcaption className="cap">
        {t(
          'Hover or click any ticket to see which features link it to others; click a feature to see which tickets share it. The “neighbours” on the right are scored by the rarity of shared features — the same idea behind the graph channel of retrieval later on. Illustrative data.',
          '悬停或点击任一工单，看它通过哪些特征连到别的工单；点击特征，看有哪些工单共享它。右侧的“邻居”按共享特征的稀有度打分，这正是后面“图关系”这一路检索的思路。示意数据。'
        )}
      </figcaption>
    </figure>
  )
}
