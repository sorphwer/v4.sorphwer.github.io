import { useCallback, useEffect, useRef, useState } from 'react'
import { T, useT } from '@/components/article/lang'
import { Tag, mulberry32, useReveal } from '../shared'

// 14 mini ticket cards scattered over the left ~40% of the box, in % of its size. Relaxed so
// cards don't overlap (y distances count 0.6× since the box is wider than tall).
const PTS = (() => {
  const rng = mulberry32(7)
  const pts = []
  for (let i = 0; i < 14; i++) pts.push({ x: 4 + rng() * 34, y: 16 + rng() * 64 })
  for (let it = 0; it < 60; it++)
    pts.forEach((a, i) =>
      pts.forEach((b, j) => {
        if (j <= i) return
        const dx = b.x - a.x
        const dy = (b.y - a.y) * 0.6
        const d = Math.hypot(dx, dy) || 0.01
        if (d < 8) {
          const q = (8 - d) / 2 / d
          a.x -= dx * q
          a.y -= (dy * q) / 0.6
          b.x += dx * q
          b.y += (dy * q) / 0.6
        }
      })
    )
  pts.forEach((p) => {
    p.x = Math.max(4, Math.min(40, p.x))
    p.y = Math.max(14, Math.min(84, p.y))
  })
  return pts
})()

// Connect near pairs. Endpoints are percentages, so the lines follow the box on resize.
const LINES = PTS.flatMap((a, i) =>
  PTS.slice(i + 1)
    .filter((b) => Math.hypot(a.x - b.x, (a.y - b.y) * 0.6) <= 15)
    .map((b) => ({ a, b }))
)

const STEPS = [
  [
    '.5s',
    [
      'Find the layer that breaks first: compare effective values and timestamps',
      '先定位是哪一层先断：比对各层生效值与时间戳',
    ],
    '#2286',
  ],
  [
    '.85s',
    [
      'Read live values in the running Pod, not just Helm values',
      '进运行中的 Pod 看实际值，不只看 Helm values',
    ],
    '#476',
  ],
  [
    '1.2s',
    [
      'Check the plugin Pod’s SDK version; upgrade if it’s hard-coded',
      '查插件 Pod 的 SDK 版本，写死的走升级',
    ],
    '#1448',
  ],
  [
    '1.55s',
    [
      'If Dify’s limits suffice, look outward: ingress, WAF, external LB',
      'Dify 侧都够了就往外查：ingress、WAF、外部 LB',
    ],
    '#2739',
  ],
]

export default function Doc() {
  const t = useT()
  const [ref, revealed] = useReveal(0.45)
  // idle → shown (cards pop in) → gathered (cards fly into the doc, which appears).
  const [phase, setPhase] = useState('idle')
  const timers = useRef([])

  const play = useCallback(() => {
    timers.current.forEach(clearTimeout)
    setPhase('idle')
    timers.current = [
      setTimeout(() => setPhase('shown'), 60),
      setTimeout(() => setPhase('gathered'), 2200),
    ]
  }, [])

  useEffect(() => {
    if (revealed) play()
  }, [revealed, play])
  useEffect(() => () => timers.current.forEach(clearTimeout), [])

  const gathered = phase === 'gathered'
  const cls = { idle: 'crys', shown: 'crys shown', gathered: 'crys shown gathered' }[phase]

  return (
    <figure className="fig wide" id="fig-doc">
      <div className="panel">
        <div ref={ref} className={cls}>
          <svg className="lines">
            {LINES.map(({ a, b }, k) => (
              <line key={k} x1={a.x + '%'} y1={a.y + '%'} x2={b.x + '%'} y2={b.y + '%'} />
            ))}
          </svg>
          <div className="clab">
            {t('14 tickets about the same kind of problem', '同一类问题的 14 张工单')}
          </div>
          <div className="after">
            <div className="big">
              14<span>→</span>1
            </div>
            <div className="sb">
              <T
                en={
                  <>
                    A dozen-plus tickets about the same thing,
                    <br />
                    distilled into one sourced troubleshooting doc.
                  </>
                }
                zh={
                  <>
                    十几张讲同一件事的工单，
                    <br />
                    收拢成一篇带出处的排查文档。
                  </>
                }
              />
            </div>
          </div>
          <div className="doc">
            <div className="row">
              <span className="lbl">{t('Troubleshooting doc', '排查文档')}</span>
              <span className="tag blue" style={{ marginLeft: 'auto' }}>
                {t('Experimental', '实验中')}
              </span>
            </div>
            <div className="t">
              {t('Plugin timeout: which layer breaks first?', '插件执行超时：先确定哪一层先断')}
            </div>
            <div className="s">
              {t('Synthesized from 14 closed tickets', '由 14 张已关闭工单合成')}
            </div>
            {STEPS.map(([delay, text, id], i) => (
              <div key={id} className="stp" style={{ transitionDelay: delay }}>
                <span className="n">{i + 1}.</span>
                <span>{t(...text)}</span>
                <Tag tone="blue" mono>
                  {id}
                </Tag>
              </div>
            ))}
            <div className="ok">
              <svg
                className="ico"
                viewBox="0 0 24 24"
                style={{ width: 18, height: 18, color: 'var(--blue)' }}
              >
                <circle cx="12" cy="12" r="9" />
                <path d="M8 12.3l2.7 2.6L16 9.5" />
              </svg>
              {t('28 citations, each one checked', '28 条出处，逐条核对')}
            </div>
          </div>
          <button className="btn replay" onClick={play}>
            {t('Replay', '再看一遍')}
          </button>
          {PTS.map((p, i) => (
            <div
              key={i}
              className="mini"
              style={{
                left: gathered ? '50%' : p.x + '%',
                top: gathered ? '32%' : p.y + '%',
                transitionDelay: i * 0.03 + 's',
              }}
            >
              <i />
              <i />
              <i style={{ width: '70%' }} />
            </div>
          ))}
        </div>
      </div>
      <figcaption className="cap">
        {t(
          'An experimental synthesized doc (steps abridged). It keeps the same discipline: every sentence must point back to the original words in some ticket.',
          '一篇实验性的合成文档（步骤有删节）。它沿用同一条纪律：每句话都要能指回某张工单里的原话。'
        )}
      </figcaption>
    </figure>
  )
}
