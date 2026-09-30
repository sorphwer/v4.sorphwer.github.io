import { useEffect, useState } from 'react'
import { useReveal } from '../shared'

const CHARTS = [
  {
    max: 1,
    title: '命中率',
    note: '排在第一的就是标注的那张工单',
    bars: [
      ['本系统', 0.711, '0.711'],
      ['开源 GraphRAG 方案', 0.57, '0.570'],
      ['工单系统自带搜索', 0, '0'],
    ],
  },
  {
    max: 26.4,
    title: '查询中位延迟',
    note: '越短越好',
    bars: [
      ['本系统', 0.8, '0.8 s'],
      ['开源 GraphRAG 方案', 1.7, '1.7 s'],
      ['工单系统自带搜索', 26.4, '26.4 s'],
    ],
  },
]
const BARS = CHARTS[0].bars.length

export default function Nums() {
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
          {CHARTS.map((ch) => (
            <div key={ch.title} className="chart">
              <h4>
                {ch.title}
                <small>{ch.note}</small>
              </h4>
              <div className="hr" />
              {ch.bars.map(([name, v, label], k) => (
                <div key={name} className={k === 0 ? 'br ours' : 'br'}>
                  <span>{name}</span>
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
              查询延迟
              <div className="sb">
                对比开源 GraphRAG 方案 · 对方开着缓存仍慢一倍，300 张同语料无缓存实测为 1/5 ·
                检索全程不调用大模型
              </div>
            </div>
          </div>
        </div>
      </div>
      <figcaption className="cap">
        同一组 149 个问题，2026-07 实测。本系统与开源 GraphRAG 方案跑在同一个 600
        张工单的语料上；工单系统自带搜索用的是它自己的全量索引。命中率只看第一名是否正确。
      </figcaption>
    </figure>
  )
}
