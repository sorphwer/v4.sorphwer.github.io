import { useEffect, useRef, useState } from 'react'
import { T, useT } from '@/components/article/lang'
import { TONE, Tag } from '../shared'

// Display strings below are [en, zh] pairs, resolved with `t` at render. Ids, versions and
// channel hit lists are shared, so both languages rank identically.
const CHS = [
  {
    k: 'sem',
    name: ['Semantic', '语义'],
    desc: ['similar meaning', '意思相近'],
    tone: 'ink',
    icon: (
      <>
        <circle cx="12" cy="12" r="2.3" fill="currentColor" />
        <circle cx="12" cy="12" r="6.5" />
        <circle cx="12" cy="12" r="10" opacity=".45" />
      </>
    ),
  },
  {
    k: 'txt',
    name: ['Full text', '原文'],
    desc: ['exact wording', '一字不差'],
    tone: 'gray',
    icon: <path d="M4 6H17M4 12H19M4 18H12" />,
  },
  {
    k: 'kw',
    name: ['Keyword', '关键词'],
    desc: ['rare keywords', '稀有关键词相同'],
    tone: 'green',
    icon: (
      <>
        <path d="M4 4H11.2L20.3 13.1Q21 13.8 20.3 14.5L14.5 20.3Q13.8 21 13.1 20.3L4 11.2Z" />
        <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor" />
      </>
    ),
  },
  {
    k: 'gr',
    name: ['Graph', '图关系'],
    desc: ['linked to top hits', '和高分工单共享冷门特征'],
    tone: 'blue',
    icon: (
      <>
        <path d="M5 18L12 5.5L19 17Z" opacity=".55" />
        <circle cx="5" cy="18" r="2.6" style={{ fill: 'var(--card)' }} />
        <circle cx="12" cy="5.5" r="2.6" style={{ fill: 'var(--card)' }} />
        <circle cx="19" cy="17" r="2.6" style={{ fill: 'var(--card)' }} />
      </>
    ),
  },
]

const PRESETS = [
  {
    label: ['Pasted error', '贴报错原文'],
    query: [
      'After upgrading to 3.9.8: panic: could not create filter · main.DifySeccomp(…)',
      '3.9.8 升级后 panic: could not create filter · main.DifySeccomp(…)',
    ],
    chips: [
      [['version 3.9.x', '版本 3.9.x'], 'blue'],
      ['seccomp', 'green'],
      ['sandbox', 'green'],
    ],
    ver: '3.9.x',
    answer: '2948',
    tickets: {
      2948: [
        ['Code execution fails after upgrading dify to 3.9.5', 'dify 升级到 3.9.5 后代码执行报错'],
        '3.9.x',
      ],
      3256: [['Sandbox unable to run in v3.9.5, …', 'Sandbox unable to run in v3.9.5, …'], '3.9.x'],
      1187: [
        [
          'Code node returns process exited with code -1',
          '代码节点返回 process exited with code -1',
        ],
        '3.8.x',
      ],
      3102: [
        [
          'seccomp: operation not permitted on ARM nodes',
          'ARM 节点上 seccomp 报 operation not permitted',
        ],
        '3.9.x',
      ],
      2410: [
        ['Plugin install fails after upgrading to 3.9.x', '升级到 3.9.x 后插件安装失败'],
        '3.9.x',
      ],
      2066: [['Code node execution times out', '代码节点执行超时'], '3.9.x'],
      931: [
        ['Sandbox network policy rejects Python requests', 'sandbox 网络策略导致 Python 请求被拒'],
        '3.9.x',
      ],
    },
    ch: {
      sem: [1187, 3256, 2948, 2066, 931],
      txt: [3102, 2948, 1187],
      kw: [3102, 2948, 3256, 2410],
      gr: [3256, 2948, 3102],
    },
  },
  {
    label: ['Symptom only', '只说症状'],
    query: [
      'Plugin calls keep timing out and the page just spins',
      '插件调用老是超时，页面一直转圈',
    ],
    chips: [
      ['plugin', 'green'],
      ['timeout', 'green'],
    ],
    ver: null,
    answer: '2739',
    tickets: {
      2739: [
        [
          'Plugin timeouts: external LB idle timeout too short',
          '插件执行超时：外部 LB 的空闲超时过短',
        ],
        '3.9.x',
      ],
      2286: [
        ['Plugin calls return 504: ingress timeout setting', '插件调用 504，ingress 超时设置'],
        '3.8.x',
      ],
      1448: [['Outdated plugin SDK makes calls hang', '插件 SDK 版本过旧导致调用挂起'], '3.9.x'],
      476: [
        [
          'Timeout changed in Helm values, not applied in the Pod',
          'Helm values 改了超时，Pod 内没生效',
        ],
        '3.8.x',
      ],
      3310: [['Workflow page stuck loading', '工作流页面一直加载中'], '3.9.x'],
      1902: [['LLM node response timeout', 'LLM 节点响应超时'], '3.9.x'],
      2555: [['Plugin daemon fails to start', 'Plugin daemon 启动失败'], '3.9.x'],
    },
    ch: {
      sem: [3310, 2739, 1902, 2286, 1448],
      txt: [1902, 3310],
      kw: [2286, 2739, 1448, 476, 2555],
      gr: [2739, 476, 2286],
    },
  },
  {
    label: ['With version', '带版本号'],
    query: [
      'After upgrading to 3.8.0, celery tasks stay pending; Redis is reachable',
      '3.8.0 升级后 celery 任务一直 pending，Redis 连得上',
    ],
    chips: [
      [['version 3.8.x', '版本 3.8.x'], 'blue'],
      ['celery', 'green'],
      ['redis', 'green'],
    ],
    ver: '3.8.x',
    answer: '1733',
    tickets: {
      1733: [
        [
          'celery worker not consuming: queue renamed in 3.8',
          'celery worker 不消费：3.8 起队列名变更',
        ],
        '3.8.x',
      ],
      3021: [
        [
          'celery tasks pending: Redis cluster mode config',
          'celery 任务 pending：Redis 集群模式配置',
        ],
        '3.9.x',
      ],
      1650: [
        [
          'Knowledge indexing stuck in queue after 3.8.0 upgrade',
          '3.8.0 升级后知识库索引卡在排队中',
        ],
        '3.8.x',
      ],
      2894: [
        ['Redis connections maxed out, tasks pile up', 'Redis 连接数打满导致任务堆积'],
        '3.9.x',
      ],
      1811: [
        ['worker replicas at 0, nothing processes tasks', 'worker 副本数为 0，任务无人处理'],
        '3.8.x',
      ],
    },
    ch: {
      sem: [3021, 1733, 1650, 2894, 1811],
      txt: [3021, 2894, 1733],
      kw: [3021, 1733, 2894, 1811],
      gr: [1733, 1650, 3021],
    },
  },
]

const ROW = 58
const TOP = 5
const SMALL_TAG = { fontSize: '10.5px', padding: '1px 5px', fontFamily: 'var(--sans)' }

const keep = (st, P, id) => !(st.ver && P.ver) || P.tickets[id][1] === P.ver

/** Weighted RRF over the enabled channels (all weights 1), best first. */
const fuse = (st, P) => {
  const sc = {}
  CHS.forEach((c) => {
    if (!st.on[c.k]) return
    P.ch[c.k]
      .filter((id) => keep(st, P, id))
      .forEach((id, i) => {
        const o = sc[id] || (sc[id] = { id: String(id), total: 0, parts: {} })
        const s = 1 / (st.k + i + 1)
        o.total += s
        o.parts[c.k] = { r: i + 1, s }
      })
  })
  return Object.values(sc).sort((a, b) => b.total - a.total || +a.id - +b.id)
}

function Channel({ c, st, P, hl, onToggle }) {
  const t = useT()
  let r = 0
  return (
    <div className={st.on[c.k] ? 'ch' : 'ch off'}>
      <button className="chh" aria-pressed={st.on[c.k]} onClick={onToggle}>
        <span className="ib" style={{ background: TONE[c.tone][1], color: TONE[c.tone][0] }}>
          <svg className="ico" viewBox="0 0 24 24">
            {c.icon}
          </svg>
        </span>
        <span>
          <div className="nm">{t(...c.name)}</div>
          <div className="ds">{t(...c.desc)}</div>
        </span>
        <span className="sw" />
      </button>
      <ol>
        {P.ch[c.k].map((id) => {
          const out = !keep(st, P, id)
          if (!out) r++
          const tk = P.tickets[id]
          const cls = [out && 'out', String(id) === P.answer && 'ans', String(id) === hl && 'hl']
          return (
            <li
              key={id}
              className={cls.filter(Boolean).join(' ')}
              title={`${t(...tk[0])} · ${tk[1]}${
                out ? t(' · version mismatch, excluded', ' · 版本不符，已排除') : ''
              }`}
            >
              <span className="r">{out ? '–' : r}</span>
              <span className="id">#{id}</span>
              <span className="tt">{t(...tk[0])}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

/** One fused row. Mounts 10px low and transparent, then slides into its slot. */
function FusedRow({ P, x, i, max, live, onHover }) {
  const t = useT()
  const ref = useRef(null)
  const [entered, setEntered] = useState(false)
  useEffect(() => {
    ref.current.getBoundingClientRect()
    setEntered(true)
  }, [])
  const tk = P.tickets[x.id]
  const cls = ['fr', i === 0 && 'first', x.id === P.answer && 'ans'].filter(Boolean).join(' ')
  const style = entered
    ? {
        transform: `translateY(${i * ROW}px)`,
        opacity: live ? 1 : 0,
        pointerEvents: live ? undefined : 'none',
      }
    : { transform: `translateY(${i * ROW + 10}px)`, opacity: 0 }
  return (
    <div
      ref={ref}
      className={cls}
      style={style}
      onMouseEnter={() => onHover(x.id)}
      onMouseLeave={() => onHover(null)}
    >
      <span className="n">{i + 1}</span>
      <div className="who">
        <div className="id">
          #{x.id}
          {x.id === P.answer && (
            <span className="tag blue" style={SMALL_TAG}>
              {t('labelled answer', '标注答案')}
            </span>
          )}
          {P.ver && tk[1] !== P.ver && (
            <span className="tag red" style={SMALL_TAG}>
              {tk[1]}
            </span>
          )}
        </div>
        <div className="tt">{t(...tk[0])}</div>
      </div>
      <div className="bar">
        {CHS.map((c) => {
          const pt = x.parts[c.k]
          return (
            <span
              key={c.k}
              style={{
                background: TONE[c.tone][0],
                width: pt ? (pt.s / max) * 100 * 0.98 + '%' : '0',
              }}
            />
          )
        })}
      </div>
      <span className="sc">{x.total.toFixed(4)}</span>
      <span className="nw">
        {Object.keys(x.parts).length}
        {t(' ch', ' 路')}
      </span>
    </div>
  )
}

/**
 * The fused ranking, remounted whenever a preset tab is clicked. A row that drops out of the
 * top keeps its last slot and content and fades out, so it can fade back in where it was.
 */
function Fused({ P, top, onHover }) {
  // id → last rendered { x, i, max }; insertion order is DOM order. Writing it during render
  // is idempotent: it depends only on `top`.
  const [seen] = useState(() => new Map())
  const max = top.length ? top[0].total : 1
  top.forEach((x, i) => seen.set(x.id, { x, i, max }))
  const live = new Set(top.map((x) => x.id))
  return (
    <div className="fused" style={{ height: TOP * ROW - 8 + 'px' }}>
      {Array.from(seen, ([id, r]) => (
        <FusedRow key={id} P={P} {...r} live={live.has(id)} onHover={onHover} />
      ))}
    </div>
  )
}

function Note({ st, P, F }) {
  const t = useT()
  const top = F.slice(0, TOP)
  if (!top.length)
    return <div className="fuse-note bad">{t('All four channels are off.', '四路都关掉了。')}</div>
  const win = top[0]
  const tk = P.tickets[win.id]
  if (win.id === P.answer) {
    const firstSomewhere = Object.values(win.parts).some((p) => p.r === 1)
    const n = Object.keys(win.parts).length
    return (
      <div className="fuse-note good">
        <T
          en={
            <>
              ✓ The top result is the labelled answer, <b>#{win.id}</b>.{' '}
            </>
          }
          zh={
            <>
              ✓ 排第一的正是标注答案 <b>#{win.id}</b>。
            </>
          }
        />
        {firstSomewhere
          ? n > 1
            ? t(`${n} channels nominated it.`, `它在 ${n} 路里都被提名。`)
            : ''
          : t(
              `It is not first in any channel, but ${n} channels nominated it.`,
              `它在任何一路都不是第一名，但 ${n} 路都提名了它。`
            )}
      </div>
    )
  }
  const ansRank = F.findIndex((x) => x.id === P.answer)
  const onCount = CHS.filter((c) => st.on[c.k]).length
  let why = ''
  if (P.ver && !st.ver && tk[1] !== P.ver)
    why = t(
      `It is a ${tk[1]} ticket, the wrong version. Try turning on the version filter above.`,
      `它是 ${tk[1]} 的工单，版本不对。打开上面的版本约束试试。`
    )
  else if (onCount < 4) why = t('Try turning on more channels.', '试着打开更多路。')
  else if (st.k < 20)
    why = t(
      "With a small k, each channel's top hit weighs too much. Try setting k back to 60.",
      'k 很小时，某一路的第一名分量太重。把 k 调回 60 试试。'
    )
  return (
    <div className="fuse-note bad">
      <T
        en={
          <>
            ✗ The top result is <b>#{win.id}</b> ({t(...tk[0])}); the labelled answer #{P.answer}{' '}
            {ansRank >= 0 ? 'ranks ' + (ansRank + 1) : 'was not found'}.{' '}
          </>
        }
        zh={
          <>
            ✗ 排第一的是 <b>#{win.id}</b>（{t(...tk[0])}），标注答案 #{P.answer}{' '}
            {ansRank >= 0 ? '排第 ' + (ansRank + 1) : '没有被找到'}。
          </>
        }
      />
      {why}
    </div>
  )
}

export default function Search() {
  const t = useT()
  const [st, setSt] = useState({
    p: 0,
    run: 0,
    on: { sem: true, txt: true, kw: true, gr: true },
    k: 60,
    ver: true,
  })
  // id of the fused row under the pointer; its entries in the channel lists get `.hl`.
  // Any other change clears it, as the original rebuilt the lists on every render.
  const [hl, setHl] = useState(null)
  const update = (patch) => {
    setSt((s) => ({ ...s, ...patch(s) }))
    setHl(null)
  }
  const P = PRESETS[st.p]
  const F = fuse(st, P)

  return (
    <figure className="fig wide" id="fig-search">
      <div className="panel wash">
        <div className="q-tabs">
          {PRESETS.map((p, i) => (
            <button
              key={i}
              className={i === st.p ? 'btn on' : 'btn'}
              onClick={() => update((s) => ({ p: i, run: s.run + 1, ver: true }))}
            >
              {t(...p.label)}
            </button>
          ))}
        </div>
        <div className="q-box">
          <svg className="ico" viewBox="0 0 24 24">
            <circle cx="10.5" cy="10.5" r="6.5" />
            <path d="M15.5 15.5L20 20" />
          </svg>
          <span className="txt">{t(...P.query)}</span>
        </div>
        <div className="q-chips">
          <span className="lbl">{t('Detected', '识别出')}</span>
          {P.chips.map((c, i) => (
            <Tag key={i} tone={c[1]} mono>
              {typeof c[0] === 'string' ? c[0] : t(...c[0])}
            </Tag>
          ))}
        </div>
        <label className={P.ver ? 'ver-row show' : 'ver-row'}>
          <input
            type="checkbox"
            checked={st.ver}
            onChange={(e) => {
              const ver = e.target.checked
              update(() => ({ ver }))
            }}
          />
          {P.ver && (
            <span>
              <T
                en={
                  <>
                    Scope by version: the query names a version, so only search <b>{P.ver}</b>{' '}
                    tickets
                  </>
                }
                zh={
                  <>
                    按版本限定范围：查询里写了版本，只在 <b>{P.ver}</b> 的工单里找
                  </>
                }
              />
            </span>
          )}
        </label>
        <div className="ch-cols">
          {CHS.map((c) => (
            <Channel
              key={c.k}
              c={c}
              st={st}
              P={P}
              hl={hl}
              onToggle={() => update((s) => ({ on: { ...s.on, [c.k]: !s.on[c.k] } }))}
            />
          ))}
        </div>
        <div className="fuse-hd">
          <span className="t">
            {t('Fused ranking', '合并排序')}{' '}
            <span className="lbl" style={{ fontWeight: 400, marginLeft: 6 }}>
              {t(
                'score = Σ 1 / (k + rank); bars are split by channel',
                '分 = Σ 1 / (k + 名次)，色条按来源分段'
              )}
            </span>
          </span>
          <label className="kctl">
            k
            <input
              type="range"
              min="1"
              max="120"
              value={st.k}
              aria-label="k"
              onChange={(e) => {
                const k = +e.target.value
                update(() => ({ k }))
              }}
            />
            <span className="k-val">{st.k}</span>
          </label>
        </div>
        <Fused key={st.run} P={P} top={F.slice(0, TOP)} onHover={setHl} />
        <Note st={st} P={P} F={F} />
      </div>
      <figcaption className="cap">
        <T
          en="Ticket ids, channel hits and ranks are illustrative; to keep it easy to follow, all four channels have weight 1. The real system weights the channels by query type (error, symptom, version-specific…), and the semantic channel itself scores two vectors separately."
          zh="票号、各路命中和名次均为示意；为了便于观察，四路权重都取 1。真实系统会按查询类型（报错、症状、版本相关……）给各路不同的权重，语义这一路内部也是两个向量各算一次。"
        />
      </figcaption>
    </figure>
  )
}
