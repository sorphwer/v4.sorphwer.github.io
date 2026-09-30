import { useEffect, useRef, useState } from 'react'
import { T, useLang, useT } from '@/components/article/lang'
import {
  G,
  KIND,
  TONE,
  clamp01,
  easeInOut,
  featR,
  lerp,
  prefersReducedMotion,
  useReveal,
} from '../shared'

const LABEL_STYLE = {
  font: '500 13px var(--sans)',
  paintOrder: 'stroke',
  stroke: 'var(--card)',
  strokeWidth: '4px',
  strokeLinejoin: 'round',
}

export default function Sea() {
  const t = useT()
  const lang = useLang()
  const [svgRef, revealed] = useReveal(0.6)
  const tagTxRef = useRef(null)
  // P = animation progress (0 scattered, 1 graph), eased towards target in the rAF loop.
  const anim = useRef({ P: 0, target: 0, t0: 0 })
  const [target, setTarget] = useState(0)
  const [frame, setFrame] = useState({ P: 0, q: 0, w: 170 })

  const snap = (now) => {
    const tx = tagTxRef.current
    setFrame({
      P: anim.current.P,
      q: ((now - anim.current.t0) / 1300) % 1,
      w: tx && tx.getComputedTextLength ? tx.getComputedTextLength() : 170,
    })
  }

  const set = (v) => {
    anim.current.target = v
    setTarget(v)
    if (prefersReducedMotion()) {
      anim.current.P = v
      snap(performance.now())
    }
  }

  useEffect(() => {
    const a = anim.current
    a.t0 = performance.now()
    snap(a.t0)
    if (prefersReducedMotion()) return
    let raf = null
    let last = null
    const loop = (now) => {
      const dt = last == null ? 0 : (now - last) / 1000
      last = now
      if (a.P !== a.target) {
        const s = dt / 2.4
        a.P = a.target > a.P ? Math.min(a.target, a.P + s) : Math.max(a.target, a.P - s)
      }
      snap(now)
      raf = requestAnimationFrame(loop)
    }
    const start = () => {
      last = null
      raf = requestAnimationFrame(loop)
    }
    if (!('IntersectionObserver' in window)) {
      start()
      return () => cancelAnimationFrame(raf)
    }
    // only animate while visible
    const io = new IntersectionObserver((es) => {
      const vis = es.some((e) => e.isIntersecting)
      if (vis && raf == null) start()
      if (!vis && raf != null) {
        cancelAnimationFrame(raf)
        raf = null
      }
    })
    io.observe(svgRef.current)
    return () => {
      io.disconnect()
      if (raf != null) cancelAnimationFrame(raf)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // The tag's text changes with the language; re-measure it even when the rAF loop is idle.
  useEffect(() => {
    snap(performance.now())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang])

  useEffect(() => {
    if (!revealed) return
    const id = setTimeout(() => {
      if (anim.current.target === 0) set(1)
    }, 1600)
    return () => clearTimeout(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [revealed])

  const { P, q, w } = frame
  const pos = G.tickets.map((tk) => {
    const m = easeInOut(clamp01((P - tk.d2 * 0.35) / 0.65))
    return [lerp(tk.x0, tk.x, m), lerp(tk.y0, tk.y, m), m]
  })
  const fo = clamp01((P - 0.45) / 0.3)
  const lo = clamp01((P - 0.8) / 0.2)
  const eo = clamp01((P - 0.6) / 0.35)
  const ro = 1 - clamp01(P / 0.25)
  const [hx, hy] = pos[0]

  return (
    <figure className="fig wide" id="fig-sea">
      <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="lab">
          {target ? (
            <T
              en={
                <>
                  <b>Linked</b>similar issues cluster
                </>
              }
              zh={
                <>
                  <b>连成一张图</b>相似的问题聚在一起
                </>
              }
            />
          ) : (
            <T
              en={
                <>
                  <b>Closed tickets</b>isolated
                </>
              }
              zh={
                <>
                  <b>已关闭的工单</b>各自孤立
                </>
              }
            />
          )}
        </div>
        <button className="btn ctl" onClick={() => set(target ? 0 : 1)}>
          {target ? t('Scatter', '打散') : t('Link them', '把它们连起来')}
        </button>
        <svg
          ref={svgRef}
          viewBox="0 0 1000 560"
          role="img"
          aria-label={t(
            'Gray dots are closed tickets; once connected by keyword, version and link, they form a graph',
            '一片灰点代表已关闭的工单，连起来后按关键词、版本和链接形成一张图'
          )}
        >
          <g>
            {G.edges.map((e, k) => {
              const f = G.feats[e.f]
              return (
                <line
                  key={k}
                  x1={pos[e.t][0]}
                  y1={pos[e.t][1]}
                  x2={f.x}
                  y2={f.y}
                  strokeWidth={1}
                  opacity={eo}
                  style={{ stroke: 'var(--edge)' }}
                />
              )
            })}
          </g>
          <g>
            {G.feats.map((f) => {
              const [ink, wash] = TONE[KIND[f.kind].tone]
              return (
                <g key={f.key} opacity={fo}>
                  <circle cx={f.x} cy={f.y} r={featR(f) + 5} style={{ fill: wash }} />
                  <circle
                    cx={f.x}
                    cy={f.y}
                    r={featR(f)}
                    strokeWidth={1.6}
                    style={{ fill: ink, stroke: 'var(--card)' }}
                  />
                </g>
              )
            })}
          </g>
          <g>
            {G.tickets.map((tk, i) => (
              <circle
                key={tk.id}
                cx={pos[i][0]}
                cy={pos[i][1]}
                r={i === 0 ? 5.2 : 4.4}
                fillOpacity={lerp(0.26, 0.8, pos[i][2])}
                style={{ fill: i === 0 && P > 0.6 ? 'var(--accent)' : 'var(--ink)' }}
              />
            ))}
          </g>
          <g>
            {G.feats.map((f) => (
              <text
                key={f.key}
                x={f.x + featR(f) + 7}
                y={f.y + 4.5}
                opacity={lo}
                style={{ ...LABEL_STYLE, fill: TONE[KIND[f.kind].tone][0] }}
              >
                {f.key}
              </text>
            ))}
          </g>
          <g>
            <circle
              cx={hx}
              cy={hy}
              r={11 + 18 * q}
              fill="none"
              strokeWidth={1}
              opacity={ro * 0.5 * (1 - q)}
              style={{ stroke: 'var(--ink)' }}
            />
            <circle
              cx={hx}
              cy={hy}
              r={11}
              fill="none"
              strokeWidth={1.6}
              opacity={ro}
              style={{ stroke: 'var(--ink)' }}
            />
            <g opacity={ro}>
              <rect
                x={hx + 18}
                y={hy - 45}
                width={w + 20}
                height={28}
                rx={6}
                style={{ fill: 'var(--card)', stroke: 'var(--hair)' }}
              />
              <text
                ref={tagTxRef}
                x={hx + 28}
                y={hy - 26}
                style={{ font: '600 13px var(--mono)', fill: 'var(--ink)' }}
              >
                {t('#2948 · closed · 6 months ago', '#2948 · 已关闭 · 半年前')}
              </text>
            </g>
          </g>
        </svg>
      </div>
      <figcaption className="cap">
        <T
          en="Each gray dot is a closed ticket; the circled one is #2948, the ticket from six months earlier in the opening film. Once they are connected by keyword (green), version (blue) and external link (purple), similar issues fall into clusters on their own. Illustrative only: neither the layout nor the counts reflect real data."
          zh="每个灰点是一张已关闭的工单，圈出来的那张就是开头短片里半年前的 #2948。按关键词（绿）、版本（蓝）、外部链接（紫）连起来之后，相似的问题自然聚成一簇。示意图，布局与数量都不代表真实数据。"
        />
      </figcaption>
    </figure>
  )
}
