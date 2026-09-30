import { Fragment, useEffect, useRef, useState } from 'react'

const SESS = ['摘要', '关键词', '链接']
const TIX = ['2948', '3256', '1187', '3102', '2410', '2066', '931', '1733']
const KEY = [
  { k: 'type', t: '调用类型' },
  { k: 'model', t: '模型名' },
  { k: 'prompt', t: 'prompt' },
  { k: 'schema', t: '输出格式' },
  { k: 'params', t: '参数' },
  { k: 'render', t: '渲染规则' },
  { k: 'pii', t: '脱敏规则' },
  { k: 'content', t: '工单内容指纹' },
  { k: 'check', t: '校验规则', off: true },
]
const ACTS = [
  {
    b: '改了“摘要”的 prompt',
    rows: [0],
    cols: 'all',
    key: ['prompt'],
    msg: '只有摘要这一类调用的版本变了；关键词和链接的缓存全部命中。',
  },
  {
    b: '#3102 有了一条新回复',
    rows: 'all',
    cols: [3],
    key: ['content'],
    msg: '只有这张工单的内容指纹变了，只重算它的三次调用。',
  },
  {
    b: '换了一个模型',
    rows: 'all',
    cols: 'all',
    key: ['model'],
    msg: '模型名是调用版本的前缀，所以三类调用全部换键、全部重算。',
  },
  {
    b: '调整了脱敏规则',
    rows: 'all',
    cols: 'all',
    key: ['pii'],
    msg: '模型看到的文本变了，结果当然可能不同，所以脱敏规则进每一类调用的版本。',
  },
  {
    b: '收紧了校验规则',
    rows: [],
    cols: [],
    key: ['check'],
    msg: '校验规则不在缓存键里。校验每次都从缓存里的原始输出重新算，一次模型调用也不需要。',
  },
]

// Cell index = row * TIX.length + col, row-major like the grid.
const CELLS = SESS.flatMap((_, r) => TIX.map((_, c) => ({ r, c })))
const hits = (a) =>
  CELLS.flatMap(({ r, c }, idx) =>
    (a.rows === 'all' || a.rows.includes(r)) && (a.cols === 'all' || a.cols.includes(c))
      ? [idx]
      : []
  )

/** Cell phase → [class suffix, glyph]. */
const PHASE = {
  idle: ['', '✓'],
  stale: [' stale', '↻'],
  run: [' stale run', '↻'],
  fresh: [' fresh', '✓'],
}

export default function Cache() {
  const [act, setAct] = useState(null)
  const [phases, setPhases] = useState(() => CELLS.map(() => 'idle'))
  const timers = useRef([])
  useEffect(() => () => timers.current.forEach(clearTimeout), [])

  const setPhase = (idx, phase) => setPhases((ps) => ps.map((p, k) => (k === idx ? phase : p)))

  const run = (i) => {
    timers.current.forEach(clearTimeout)
    timers.current = []
    const hit = hits(ACTS[i])
    const hitSet = new Set(hit)
    setAct(i)
    setPhases(CELLS.map((_, idx) => (hitSet.has(idx) ? 'stale' : 'idle')))
    hit.forEach((idx, k) => {
      timers.current.push(setTimeout(() => setPhase(idx, 'run'), 500 + k * 70))
      timers.current.push(setTimeout(() => setPhase(idx, 'fresh'), 800 + k * 70))
    })
  }

  const a = act == null ? null : ACTS[act]

  return (
    <figure className="fig wide" id="fig-cache">
      <div className="panel">
        <div className="cache">
          <div>
            <div className="cgrid">
              <span />
              {TIX.map((t) => (
                <span key={t} className="chd">
                  #{t}
                </span>
              ))}
              {SESS.map((s, r) => (
                <Fragment key={s}>
                  <span className="rh">{s}</span>
                  {TIX.map((t, c) => {
                    const [cls, glyph] = PHASE[phases[r * TIX.length + c]]
                    return (
                      <span key={t} className={'cell' + cls}>
                        {glyph}
                      </span>
                    )
                  })}
                </Fragment>
              ))}
            </div>
            <div className="ckey">
              <span className="lbl">缓存键</span>
              <div className="parts">
                {KEY.map((p, i) => (
                  <Fragment key={p.k}>
                    {i === 1 && <span className="lbl">调用版本 = hash(</span>}
                    <span className={p.off ? 'p off' : a && a.key.includes(p.k) ? 'p hit' : 'p'}>
                      {p.t}
                    </span>
                    {i === 6 && <span className="lbl">)</span>}
                  </Fragment>
                ))}
              </div>
            </div>
          </div>
          <div>
            <div className="lbl" style={{ marginBottom: 8 }}>
              如果我们……
            </div>
            <div className="cacts">
              {ACTS.map((x, i) => (
                <button key={x.b} className={i === act ? 'btn on' : 'btn'} onClick={() => run(i)}>
                  {x.b}
                </button>
              ))}
            </div>
            <div className="ccount">
              <b>{a ? hits(a).length : 0}</b>
              <span>次模型调用 / 共 24 格</span>
            </div>
            <div className="cmsg">{a ? a.msg : '选一种改动，看哪些缓存会失效。'}</div>
          </div>
        </div>
      </div>
      <figcaption className="cap">
        每一格是一张工单的一次模型调用（摘要、关键词、链接三类）。绿色是缓存命中，橙色是需要重新调用模型。
      </figcaption>
    </figure>
  )
}
