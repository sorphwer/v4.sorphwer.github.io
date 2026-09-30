import { Fragment, useEffect, useRef, useState } from 'react'
import { useLang, useT } from '@/components/article/lang'

// Row = call type. English labels run 1px smaller (see `rhStyle`) to fit the 58px header
// column, and carry soft hyphens so they break inside the narrower mobile column instead of
// spilling into the cells.
const SESS = [
  ['Sum\u00admary', '摘要'],
  ['Key\u00adwords', '关键词'],
  ['Links', '链接'],
]
const TIX = ['2948', '3256', '1187', '3102', '2410', '2066', '931', '1733']
const KEY = [
  { k: 'type', t: ['call type', '调用类型'] },
  { k: 'model', t: ['model', '模型名'] },
  { k: 'prompt', t: ['prompt', 'prompt'] },
  { k: 'schema', t: ['output schema', '输出格式'] },
  { k: 'params', t: ['params', '参数'] },
  { k: 'render', t: ['render rules', '渲染规则'] },
  { k: 'pii', t: ['masking rules', '脱敏规则'] },
  { k: 'content', t: ['content fingerprint', '工单内容指纹'] },
  { k: 'check', t: ['validation rules', '校验规则'], off: true },
]
const ACTS = [
  {
    b: ['change the summary prompt', '改了“摘要”的 prompt'],
    rows: [0],
    cols: 'all',
    key: ['prompt'],
    msg: [
      'Only the summary call’s version changes; keyword and link caches all still hit.',
      '只有摘要这一类调用的版本变了；关键词和链接的缓存全部命中。',
    ],
  },
  {
    b: ['get a new reply on #3102', '#3102 有了一条新回复'],
    rows: 'all',
    cols: [3],
    key: ['content'],
    msg: [
      'Only this ticket’s content fingerprint changes, so only its three calls rerun.',
      '只有这张工单的内容指纹变了，只重算它的三次调用。',
    ],
  },
  {
    b: ['switch to another model', '换了一个模型'],
    rows: 'all',
    cols: 'all',
    key: ['model'],
    msg: [
      'The model name prefixes every call version, so all three call types get new keys and all rerun.',
      '模型名是调用版本的前缀，所以三类调用全部换键、全部重算。',
    ],
  },
  {
    b: ['adjust the masking rules', '调整了脱敏规则'],
    rows: 'all',
    cols: 'all',
    key: ['pii'],
    msg: [
      'The text the model sees changes, so its output may too; that’s why masking rules are part of every call version.',
      '模型看到的文本变了，结果当然可能不同，所以脱敏规则进每一类调用的版本。',
    ],
  },
  {
    b: ['tighten the validation rules', '收紧了校验规则'],
    rows: [],
    cols: [],
    key: ['check'],
    msg: [
      'Validation rules aren’t in the cache key. Validation reruns on the cached raw output every time, without a single model call.',
      '校验规则不在缓存键里。校验每次都从缓存里的原始输出重新算，一次模型调用也不需要。',
    ],
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
  const t = useT()
  const rhStyle = useLang() === 'en' ? { fontSize: 12 } : undefined
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
              {TIX.map((id) => (
                <span key={id} className="chd">
                  #{id}
                </span>
              ))}
              {SESS.map((s, r) => (
                <Fragment key={r}>
                  <span className="rh" style={rhStyle}>
                    {t(...s)}
                  </span>
                  {TIX.map((id, c) => {
                    const [cls, glyph] = PHASE[phases[r * TIX.length + c]]
                    return (
                      <span key={id} className={'cell' + cls}>
                        {glyph}
                      </span>
                    )
                  })}
                </Fragment>
              ))}
            </div>
            <div className="ckey">
              <span className="lbl">{t('Cache key', '缓存键')}</span>
              <div className="parts">
                {KEY.map((p, i) => (
                  <Fragment key={p.k}>
                    {i === 1 && (
                      <span className="lbl">{t('call version = hash(', '调用版本 = hash(')}</span>
                    )}
                    <span className={p.off ? 'p off' : a && a.key.includes(p.k) ? 'p hit' : 'p'}>
                      {t(...p.t)}
                    </span>
                    {i === 6 && <span className="lbl">)</span>}
                  </Fragment>
                ))}
              </div>
            </div>
          </div>
          <div>
            <div className="lbl" style={{ marginBottom: 8 }}>
              {t('What if we…', '如果我们……')}
            </div>
            <div className="cacts">
              {ACTS.map((x, i) => (
                <button key={i} className={i === act ? 'btn on' : 'btn'} onClick={() => run(i)}>
                  {t(...x.b)}
                </button>
              ))}
            </div>
            <div className="ccount">
              <b>{a ? hits(a).length : 0}</b>
              <span>{t('model calls / 24 cells', '次模型调用 / 共 24 格')}</span>
            </div>
            <div className="cmsg">
              {a
                ? t(...a.msg)
                : t(
                    'Pick a change to see which cache entries go stale.',
                    '选一种改动，看哪些缓存会失效。'
                  )}
            </div>
          </div>
        </div>
      </div>
      <figcaption className="cap">
        {t(
          'Each cell is one model call for one ticket (three kinds: summary, keywords, links). Green is a cache hit; orange means the model must be called again.',
          '每一格是一张工单的一次模型调用（摘要、关键词、链接三类）。绿色是缓存命中，橙色是需要重新调用模型。'
        )}
      </figcaption>
    </figure>
  )
}
