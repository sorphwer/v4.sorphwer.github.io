import { useEffect, useRef, useState } from 'react'
import { Tag, prefersReducedMotion } from '../shared'

const CM = [
  {
    id: '…5901',
    who: '[NAME_1]',
    role: '客户',
    body: '升级到 3.9.5 之后，代码节点全部报错：process exited with code -1，日志里能看到 main.DifySeccomp(…)。',
  },
  {
    id: '…5924',
    who: '支持工程师',
    role: '支持',
    body: '当前错误发生在 sandbox 初始化 DifySeccomp 阶段，不是代码逻辑本身的问题。建议先确认 sandbox 镜像版本和 Dify 版本是否配套。',
  },
  {
    id: '…5988',
    who: '支持工程师',
    role: '支持',
    body: '请把 dify-sandbox 镜像升级到与 3.9.5 配套的版本，然后重启 sandbox。',
  },
  {
    id: '…6010',
    who: '[NAME_1]',
    role: '客户',
    body: '升级 sandbox 镜像并重启之后，代码节点恢复正常了，谢谢。',
  },
]
const REAL = [
  {
    en: 'The failure occurred during Sandbox DifySeccomp initialization, not in the customer’s Python logic.',
    cid: '…5924',
    q: '当前错误发生在 sandbox 初始化 DifySeccomp 阶段，不是代码逻辑本身的问题。',
  },
  {
    en: 'Support advised upgrading the dify-sandbox image to the version matching 3.9.5 and restarting it.',
    cid: '…5988',
    q: '请把 dify-sandbox 镜像升级到与 3.9.5 配套的版本，然后重启 sandbox。',
  },
  {
    en: 'The customer confirmed code nodes worked again after the upgrade.',
    cid: '…6010',
    q: '升级 sandbox 镜像并重启之后，代码节点恢复正常了',
  },
]
const FAKE = {
  en: 'The root cause was an insufficient memory limit on the sandbox pod.',
  cid: '…5924',
  q: 'sandbox Pod 的内存限额设置过低',
  fake: true,
}
const RESOLVED = { cid: '…6010', q: '代码节点恢复正常了' }
const RESOLVED_OK = CM.some((m) => m.id === RESOLVED.cid && m.body.includes(RESOLVED.q))

const norm = (s) => s.replace(/\s+/g, ' ').trim()
// The check really runs: each quote must appear verbatim in the comment it cites.
const found = (c) => {
  const m = CM.find((x) => x.id === c.cid)
  return !!m && norm(m.body).includes(norm(c.q))
}

const INIT = { fake: false, phase: 'idle', sel: 0 }

const Mk = ({ s }) => (
  <span className={`mk ${s}`}>{s === 'ok' ? '✓' : s === 'bad' ? '✗' : '!'}</span>
)

function Body({ text, q }) {
  const k = q ? text.indexOf(q) : -1
  if (k < 0) return text
  return (
    <>
      {text.slice(0, k)}
      <mark>{q}</mark>
      {text.slice(k + q.length)}
    </>
  )
}

export default function Ground() {
  const [st, setSt] = useState(INIT)
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])

  const claims = st.fake ? [REAL[0], FAKE, REAL[1], REAL[2]] : REAL
  const cur = claims[st.sel]
  const nOk = claims.filter(found).length
  const allOk = nOk === claims.length
  const q = allOk ? 'ok' : st.phase === 'degraded' ? 'warn' : 'bad'
  const qTxt = allOk
    ? `原话逐字可查：${nOk} / ${claims.length}`
    : st.phase === 'degraded'
    ? `原话逐字可查：${nOk} / ${claims.length} · 1 句降级为“未核实”`
    : st.phase === 'repairing'
    ? '修复中：只重写引用部分，摘要冻结…'
    : `原话逐字可查：${nOk} / ${claims.length} · 需要修复`

  const addFake = () => setSt({ fake: true, phase: 'idle', sel: 1 })
  const repair = () => {
    setSt((s) => ({ ...s, phase: 'repairing' }))
    timer.current = setTimeout(
      () => setSt((s) => ({ ...s, phase: 'degraded' })),
      prefersReducedMotion() ? 0 : 1300
    )
  }
  const reset = () => {
    clearTimeout(timer.current)
    setSt(INIT)
  }

  return (
    <figure className="fig wide" id="fig-ground">
      <div className="panel">
        <div className="gr">
          <div>
            <div className="lbl" style={{ marginBottom: 8 }}>
              工单 #2948 的对话（脱敏后）
            </div>
            <div className="conv">
              {CM.map((m) => {
                const focus = !!cur && cur.cid === m.id
                const ok = focus && found(cur)
                const miss = focus && !ok
                return (
                  <div className={`cm${focus ? ' focus' : ''}${miss ? ' miss' : ''}`} key={m.id}>
                    <div className="meta">
                      <span>comment_id={m.id}</span>
                      <Tag tone={m.role === '客户' ? 'gray' : 'blue'}>{m.role}</Tag>
                      <span>{m.who}</span>
                    </div>
                    <div className="body">
                      <Body text={m.body} q={ok ? cur.q : null} />
                    </div>
                    {miss ? (
                      <div className="lbl" style={{ color: 'var(--pink)', marginTop: 4 }}>
                        这条回复里找不到这句原话
                      </div>
                    ) : null}
                  </div>
                )
              })}
            </div>
          </div>
          <div>
            <div className="lbl" style={{ marginBottom: 8 }}>
              解决方案摘要，拆成结论 · 点击查看原话
            </div>
            <div className="claims">
              {claims.map((c, i) => {
                const ok = found(c)
                const ung = !ok && st.phase === 'degraded'
                const cls = [
                  'cl',
                  i === st.sel && 'on',
                  // mounted only when the model invents it, so the pop-in plays exactly once
                  c.fake && 'fake enter',
                  ung && 'ungrounded',
                ]
                return (
                  <button
                    className={cls.filter(Boolean).join(' ')}
                    key={c.en}
                    onClick={() => setSt((s) => ({ ...s, sel: i }))}
                  >
                    <div className="en">{c.en}</div>
                    <div className="src">
                      {ok ? (
                        <>
                          <span style={{ color: 'var(--blue)' }}>✓ 原话</span> {c.cid}
                        </>
                      ) : ung ? (
                        <>
                          <span className="tag gray" style={{ fontSize: 11 }}>
                            未核实
                          </span>{' '}
                          保留，但不被当作可信结论
                        </>
                      ) : (
                        <>
                          <span style={{ color: 'var(--pink)' }}>✗ 找不到原话</span> 声称来自{' '}
                          {c.cid}
                        </>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
            <div className="checks">
              <div>
                <Mk s="ok" />
                每句结论都出自摘要，拼起来覆盖整段摘要
              </div>
              <div>
                {st.phase === 'repairing' ? <span className="spin" /> : <Mk s={q} />}
                {qTxt}
              </div>
              <div>
                <Mk s={RESOLVED_OK ? 'ok' : 'bad'} />
                标为“已解决”，并有客户确认的原话（{RESOLVED.cid}）
              </div>
            </div>
            <div className="gr-ctl">
              <button className="btn" disabled={st.fake} onClick={addFake}>
                让模型编一句
              </button>
              <button
                className="btn primary"
                disabled={!(st.fake && st.phase === 'idle')}
                onClick={repair}
              >
                修复一轮
              </button>
              <button className="btn" style={{ marginLeft: 'auto' }} onClick={reset}>
                重置
              </button>
            </div>
          </div>
        </div>
      </div>
      <figcaption className="cap">
        校验是真的在跑：右边每句结论的“原话”，都在左边对应回复里逐字查找。第一句结论和它的原话来自真实工单
        #2948，其余对话为说明而改写。
      </figcaption>
    </figure>
  )
}
