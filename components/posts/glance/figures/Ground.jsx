import { useEffect, useRef, useState } from 'react'
import { T, useLang, useT } from '@/components/article/lang'
import { Tag, prefersReducedMotion } from '../shared'

// Bodies and quotes come per language ({ en, zh }); every quote is a verbatim
// substring of the same-language body it cites, so the check below holds in both.
const CM = [
  {
    id: '…5901',
    cust: true,
    body: {
      en: 'After upgrading to 3.9.5, every code node fails: process exited with code -1, and the logs show main.DifySeccomp(…).',
      zh: '升级到 3.9.5 之后，代码节点全部报错：process exited with code -1，日志里能看到 main.DifySeccomp(…)。',
    },
  },
  {
    id: '…5924',
    body: {
      en: 'The error occurs while the sandbox initializes DifySeccomp, not in your code logic. Please first check that your sandbox image version matches your Dify version.',
      zh: '当前错误发生在 sandbox 初始化 DifySeccomp 阶段，不是代码逻辑本身的问题。建议先确认 sandbox 镜像版本和 Dify 版本是否配套。',
    },
  },
  {
    id: '…5988',
    body: {
      en: 'Please upgrade the dify-sandbox image to the version that matches 3.9.5, then restart the sandbox.',
      zh: '请把 dify-sandbox 镜像升级到与 3.9.5 配套的版本，然后重启 sandbox。',
    },
  },
  {
    id: '…6010',
    cust: true,
    body: {
      en: 'After upgrading the sandbox image and restarting, code nodes work normally again. Thanks!',
      zh: '升级 sandbox 镜像并重启之后，代码节点恢复正常了，谢谢。',
    },
  },
]
const REAL = [
  {
    en: 'The failure occurred during Sandbox DifySeccomp initialization, not in the customer’s Python logic.',
    cid: '…5924',
    q: {
      en: 'The error occurs while the sandbox initializes DifySeccomp, not in your code logic.',
      zh: '当前错误发生在 sandbox 初始化 DifySeccomp 阶段，不是代码逻辑本身的问题。',
    },
  },
  {
    en: 'Support advised upgrading the dify-sandbox image to the version matching 3.9.5 and restarting it.',
    cid: '…5988',
    q: {
      en: 'Please upgrade the dify-sandbox image to the version that matches 3.9.5, then restart the sandbox.',
      zh: '请把 dify-sandbox 镜像升级到与 3.9.5 配套的版本，然后重启 sandbox。',
    },
  },
  {
    en: 'The customer confirmed code nodes worked again after the upgrade.',
    cid: '…6010',
    q: {
      en: 'After upgrading the sandbox image and restarting, code nodes work normally again',
      zh: '升级 sandbox 镜像并重启之后，代码节点恢复正常了',
    },
  },
]
const FAKE = {
  en: 'The root cause was an insufficient memory limit on the sandbox pod.',
  cid: '…5924',
  q: {
    en: 'the memory limit on the sandbox pod is set too low',
    zh: 'sandbox Pod 的内存限额设置过低',
  },
  fake: true,
}
const RESOLVED = {
  cid: '…6010',
  q: { en: 'code nodes work normally again', zh: '代码节点恢复正常了' },
}

const norm = (s) => s.replace(/\s+/g, ' ').trim()
// The check really runs: each quote must appear verbatim in the comment it cites.
const found = (c, lang) => {
  const m = CM.find((x) => x.id === c.cid)
  return !!m && norm(m.body[lang]).includes(norm(c.q[lang]))
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
  const t = useT()
  const lang = useLang()
  const [st, setSt] = useState(INIT)
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])

  const claims = st.fake ? [REAL[0], FAKE, REAL[1], REAL[2]] : REAL
  const cur = claims[st.sel]
  const nOk = claims.filter((c) => found(c, lang)).length
  const resolvedOk = found(RESOLVED, lang)
  const allOk = nOk === claims.length
  const q = allOk ? 'ok' : st.phase === 'degraded' ? 'warn' : 'bad'
  const tally = t(
    `Quotes found verbatim: ${nOk} / ${claims.length}`,
    `原话逐字可查：${nOk} / ${claims.length}`
  )
  const qTxt = allOk
    ? tally
    : st.phase === 'degraded'
    ? `${tally}${t(' · 1 claim marked “unverified”', ' · 1 句降级为“未核实”')}`
    : st.phase === 'repairing'
    ? t('Repairing: rewriting citations only, summary frozen…', '修复中：只重写引用部分，摘要冻结…')
    : `${tally}${t(' · needs repair', ' · 需要修复')}`

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
              <T en="Ticket #2948 conversation (redacted)" zh="工单 #2948 的对话（脱敏后）" />
            </div>
            <div className="conv">
              {CM.map((m) => {
                const focus = !!cur && cur.cid === m.id
                const ok = focus && found(cur, lang)
                const miss = focus && !ok
                return (
                  <div className={`cm${focus ? ' focus' : ''}${miss ? ' miss' : ''}`} key={m.id}>
                    <div className="meta">
                      <span>comment_id={m.id}</span>
                      <Tag tone={m.cust ? 'gray' : 'blue'}>
                        {m.cust ? t('Customer', '客户') : t('Support', '支持')}
                      </Tag>
                      <span>{m.cust ? '[NAME_1]' : t('Support engineer', '支持工程师')}</span>
                    </div>
                    <div className="body">
                      <Body text={m.body[lang]} q={ok ? cur.q[lang] : null} />
                    </div>
                    {miss ? (
                      <div className="lbl" style={{ color: 'var(--pink)', marginTop: 4 }}>
                        <T
                          en="This quote does not appear in this reply"
                          zh="这条回复里找不到这句原话"
                        />
                      </div>
                    ) : null}
                  </div>
                )
              })}
            </div>
          </div>
          <div>
            <div className="lbl" style={{ marginBottom: 8 }}>
              <T
                en="Solution summary, split into claims · click one to see its quote"
                zh="解决方案摘要，拆成结论 · 点击查看原话"
              />
            </div>
            <div className="claims">
              {claims.map((c, i) => {
                const ok = found(c, lang)
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
                          <span style={{ color: 'var(--blue)' }}>{t('✓ quote', '✓ 原话')}</span>{' '}
                          {c.cid}
                        </>
                      ) : ung ? (
                        <>
                          <span className="tag gray" style={{ fontSize: 11 }}>
                            {t('unverified', '未核实')}
                          </span>{' '}
                          {t(
                            'kept, but not treated as a trusted claim',
                            '保留，但不被当作可信结论'
                          )}
                        </>
                      ) : (
                        <>
                          <span style={{ color: 'var(--pink)' }}>
                            {t('✗ quote not found', '✗ 找不到原话')}
                          </span>{' '}
                          {t('claims to be from', '声称来自')} {c.cid}
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
                {t(
                  'Claims come from the summary and cover all of it',
                  '每句结论都出自摘要，拼起来覆盖整段摘要'
                )}
              </div>
              <div>
                {st.phase === 'repairing' ? <span className="spin" /> : <Mk s={q} />}
                {qTxt}
              </div>
              <div>
                <Mk s={resolvedOk ? 'ok' : 'bad'} />
                {t(
                  `Marked “resolved”: customer confirmed (${RESOLVED.cid})`,
                  `标为“已解决”，并有客户确认的原话（${RESOLVED.cid}）`
                )}
              </div>
            </div>
            <div className="gr-ctl">
              <button className="btn" disabled={st.fake} onClick={addFake}>
                {t('Invent one', '让模型编一句')}
              </button>
              <button
                className="btn primary"
                disabled={!(st.fake && st.phase === 'idle')}
                onClick={repair}
              >
                {t('Repair', '修复一轮')}
              </button>
              <button className="btn" style={{ marginLeft: 'auto' }} onClick={reset}>
                {t('Reset', '重置')}
              </button>
            </div>
          </div>
        </div>
      </div>
      <figcaption className="cap">
        <T
          en="The check really runs: each claim’s quote on the right is searched for verbatim in the reply it cites on the left. The first claim and its quote come from real ticket #2948 (translated here); the rest of the conversation was rewritten for illustration."
          zh={
            <>
              校验是真的在跑：右边每句结论的“原话”，都在左边对应回复里逐字查找。第一句结论和它的原话来自真实工单
              #2948，其余对话为说明而改写。
            </>
          }
        />
      </figcaption>
    </figure>
  )
}
