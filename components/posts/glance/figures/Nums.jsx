import { useEffect, useState } from 'react'
import { T, useT } from '@/components/article/lang'
import { useReveal } from '../shared'

// Display strings are [en, zh] pairs, resolved with `t` at render.
const CHARTS = [
  {
    max: 1,
    title: ['Hit rate', '命中率'],
    note: ['top result is the labelled ticket', '排在第一的就是标注的那张工单'],
    bars: [
      [['This system', '本系统'], 0.711, '0.711'],
      [['OSS GraphRAG', '开源 GraphRAG 方案'], 0.57, '0.570'],
      [['Built-in ticket search', '工单系统自带搜索'], 0, '0'],
    ],
  },
  {
    max: 26.4,
    title: ['Median query latency', '查询中位延迟'],
    note: ['lower is better', '越短越好'],
    bars: [
      [['This system', '本系统'], 0.8, '0.8 s'],
      [['OSS GraphRAG', '开源 GraphRAG 方案'], 1.7, '1.7 s'],
      [['Built-in ticket search', '工单系统自带搜索'], 26.4, '26.4 s'],
    ],
  },
]
const BARS = CHARTS[0].bars.length

export default function Nums() {
  const t = useT()
  const [ref, revealed] = useReveal(0.35)
  // Bars grow one row at a time (the same row in both charts together) once revealed.
  const [grown, setGrown] = useState(0)
  useEffect(() => {
    if (!revealed) return
    const timers = Array.from({ length: BARS }, (_, k) =>
      setTimeout(() => setGrown(k + 1), 150 + k * 250)
    )
    return () => timers.forEach(clearTimeout)
  }, [revealed])

  return (
    <figure ref={ref} className={revealed ? 'fig wide in' : 'fig wide'} id="fig-nums">
      <div className="panel">
        <div className="nums">
          {CHARTS.map((ch, c) => (
            <div key={c} className="chart">
              <h4>
                {t(...ch.title)}
                <small>{t(...ch.note)}</small>
              </h4>
              <div className="hr" />
              {ch.bars.map(([name, v, label], k) => (
                <div key={k} className={k === 0 ? 'br ours' : 'br'}>
                  <span>{t(...name)}</span>
                  <div className="track">
                    <i
                      style={
                        k < grown
                          ? { width: (v > 0 ? Math.max(3, (v / ch.max) * 100) : 0) + '%' }
                          : undefined
                      }
                    />
                  </div>
                  <span className="v">{label}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="tiles">
          <div className="tile">
            <span className="big">1/2</span>
            <div className="tx">
              <T en="Query latency" zh="查询延迟" />
              <div className="sb">
                <T
                  en="vs. open-source GraphRAG · it is still 2× slower with its cache on; uncached on 300 tickets of the same corpus, ours takes 1/5 the time · no LLM calls anywhere in retrieval"
                  zh="对比开源 GraphRAG 方案 · 对方开着缓存仍慢一倍，300 张同语料无缓存实测为 1/5 · 检索全程不调用大模型"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      <figcaption className="cap">
        <T
          en="The same 149 questions, measured 2026-07. This system and the open-source GraphRAG setup ran on the same 600-ticket corpus; the ticket system's built-in search used its own full index. Hit rate only counts whether the top result is correct."
          zh="同一组 149 个问题，2026-07 实测。本系统与开源 GraphRAG 方案跑在同一个 600 张工单的语料上；工单系统自带搜索用的是它自己的全量索引。命中率只看第一名是否正确。"
        />
      </figcaption>
    </figure>
  )
}
