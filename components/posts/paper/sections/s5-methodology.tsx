import type { ReactNode } from 'react'

import { C, DiagramPanel, Fig, Lead, M, MD, P, Section, T } from '../shared'

/** One stage card in the golden-query mining flow (Figure 8). */
function Stage({
  label,
  emphasis = false,
  children,
}: {
  label: string
  emphasis?: boolean
  children: ReactNode
}) {
  return (
    <div
      className={`flex-1 min-w-[130px] rounded-md px-3.5 py-2.5 ${
        emphasis
          ? 'bg-ds-gray-1000 text-ds-background-100'
          : 'bg-ds-background-100 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]'
      }`}
    >
      <div
        className={`font-geist-mono text-[10.5px] ${emphasis ? 'opacity-70' : 'text-ds-gray-700'}`}
      >
        {label}
      </div>
      {children}
    </div>
  )
}

/** Centered right-arrow separator between mining stages. */
function Arrow() {
  return <span className="self-center text-ds-gray-600">&rarr;</span>
}

export default function S5Methodology() {
  return (
    <Section id="s5" num="05" en="Evaluation Method" zh="评测方法">
      <P
        en={
          <>
            <Lead>Evaluation principles.</Lead> The benchmark sets the unit of evaluation to
            ticket-level top-
            <M t="k" /> membership and rank for each golden query. Correspondingly, the ground truth
            uses the ticket-level label <C>expected_ticket_ids</C> and ticket-level scoring; labels
            are frozen outside the algorithm, grounded in quantities that do not drift with the RRF
            score, confidence, or algorithm version. The number of metrics stays restrained so that
            every metric can be interpreted directly.
          </>
        }
        zh={
          <>
            <Lead>评测原则。</Lead>Benchmark 将评测单位定为每条 golden query 的 ticket 级 top-
            <M t="k" /> 进入与名次。对应地，ground truth 采用 ticket 级标签{' '}
            <C>expected_ticket_ids</C>，采用 ticket 级评分；标签在算法之外冻结，以不随 RRF
            分数、confidence
            或算法版本漂移的量作为依据；指标数量保持克制，保证每个指标都能直接解释。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Data source and freezing.</Lead> The candidates all come from D1{' '}
            <C>retrieval_calls</C> — that is, real agent traffic rather than synthetic queries. The
            mining chain is deterministic, with a fixed random seed <C>20260704</C>:
          </>
        }
        zh={
          <>
            <Lead>数据来源与冻结。</Lead>候选全部来自 D1 <C>retrieval_calls</C>
            ，也就是真实 agent 流量，而不是合成查询。挖掘链路是确定性的，固定随机种子{' '}
            <C>20260704</C>：
          </>
        }
      />

      <Fig
        num={8}
        en={
          <>
            The full golden-query pipeline: mining &rarr; annotation &rarr; freezing &rarr;
            write-back. Four gates — G0 (validity) / G1 (excludes direct ticket-ID lookups,{' '}
            <C>top_rrf&lt;0.999</C>) / G2 (normalized dedup) / G3 (stratification) — yield 214
            candidates. Adapted from Dashboard doc §5.7.
          </>
        }
        zh={
          <>
            golden query 挖掘 &rarr; 标注 &rarr; 冻结 &rarr; 回写全流程。四道闸门 G0（有效性）/
            G1（排工单号直查 <C>top_rrf&lt;0.999</C>）/ G2（归一化去重）/ G3（分层）产出 214
            候选。改编自 Dashboard 文档 §5.7。
          </>
        }
      >
        <DiagramPanel>
          <div className="flex flex-wrap items-stretch gap-2">
            <Stage label="SOURCE">
              <div className="text-[13px] font-medium">D1 retrieval_calls</div>
              <div className="text-[13px] text-ds-gray-900">
                <T en="real agent traffic" zh="真实 agent 流量" />
              </div>
            </Stage>
            <Arrow />
            <Stage label="FETCH">
              <div className="text-[13px] font-medium">
                <T en="914 rows" zh="914 行" />
              </div>
            </Stage>
            <Arrow />
            <Stage label="GATE G0–G3">
              <div className="text-[13px] font-medium">
                <T en="214 candidates" zh="214 候选" />
              </div>
            </Stage>
            <Arrow />
            <Stage label="HYDRATE">
              <div className="text-[13px] text-ds-gray-900">
                <T en="attach served top-k hits" zh="附当时 served top-k hits" />
              </div>
            </Stage>
            <Arrow />
            <Stage label="JUDGE">
              <div className="text-[13px] text-ds-gray-900">
                <T en="LLM-assisted + manual spot-check ≥15%" zh="LLM 辅助 + 人工抽检 ≥15%" />
              </div>
              <div className="text-[13px] font-medium">149 found / 62 missing / 3 junk</div>
            </Stage>
            <Arrow />
            <Stage label="FREEZE" emphasis>
              <div className="text-[13px] font-medium">
                <T en="149 regression" zh="149 regression" />
              </div>
              <div className="font-geist-mono text-[10.5px] opacity-80">
                v2-2026-07-05-golden.yaml · ticket_count=1032
              </div>
            </Stage>
          </div>
          <div className="mt-2.5 text-[13px] text-ds-gray-900">
            <T
              en={
                <>
                  The build also runs <C>mark</C>: it writes back <C>golden_id</C> (a traceability
                  marker only, distinct from ground truth).
                </>
              }
              zh={
                <>
                  build 同时执行 <C>mark</C>：回写 <C>golden_id</C>
                  （仅溯源，非真值）。
                </>
              }
            />
          </div>
        </DiagramPanel>
      </Fig>

      <P
        en={
          <>
            During annotation, samples that already have the answer among the hits are recorded as{' '}
            <C>verdict: found</C>; those that should exist in the store but were not recalled are
            marked <C>missing</C>, a common source of the headroom layer (samples that should be in
            the store but have not yet been recalled); meaningless queries are judged <C>junk</C>{' '}
            and discarded. This step yields 149 found, 62 missing, and 3 junk. The evaluation set is
            frozen only after the 23/149 stratified spot-check samples have been reviewed. The final
            Golden Query contains 149 regression queries, all rewritten by an LLM into natural
            language and free of version constraints, so they all take the global path.
          </>
        }
        zh={
          <>
            标注时，hits 中已有答案的样本记为 <C>verdict: found</C>
            ；库里应有但未召回的记为 <C>missing</C>
            ，这是 headroom 层（库中应有而尚未召回的样本）的常见来源；无意义查询判为 <C>
              junk
            </C>{' '}
            并丢弃。这一步产出 149 found、62 missing、3 junk。23/149
            的分层抽检样本审完后，评测集才冻结。最终 Golden Query 包含 149 条 regression
            查询，全部由 LLM 改写为自然语言，且不含版本约束，因此都走全局路径。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Metric definitions.</Lead> The metrics are based on binary relevance; the
            implementation is in <C>metrics.py</C>. Denote the rank of the first hit as{' '}
            <M t="r^*" />:
          </>
        }
        zh={
          <>
            <Lead>指标定义。</Lead>指标基于二值相关性，实现见 <C>metrics.py</C>
            。记首个命中的排名为 <M t="r^*" />：
          </>
        }
      />
      <MD t="\text{Hit@}k=\mathbb{1}[r^*\le k],\qquad \text{MRR@}k=\frac{1}{r^*}\,\mathbb{1}[r^*\le k]" />
      <MD t="\text{nDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k},\quad \text{DCG@}k=\!\!\sum_{i:\,t_i\in E,\,i\le k}\!\!\frac{1}{\log_2(i+1)},\quad \text{IDCG@}k=\!\!\sum_{i=1}^{\min(|E|,k)}\!\!\frac{1}{\log_2(i+1)}" />
      <P
        en={
          <>
            A hit scores gain=1 and a miss 0; IDCG takes the top <M t="\min(|E|,k)" /> ideal
            positions. The metrics stay simple so that the evaluation conclusions do not depend on
            some complex scoring convention.
          </>
        }
        zh={
          <>
            命中记 gain=1，未命中记 0，IDCG 取 <M t="\min(|E|,k)" />
            个理想位置。指标保持简单，是为了让评测结论不依赖某个复杂打分口径。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Ablation design.</Lead> The five setups form a 2&times;2 of kw&times;gr plus an
            external anchor: base (iv/sv/ft), +kw, +gr (kw off), +kw+gr (the full pipeline), and
            Zendesk native search. Adding the off-diagonal +gr cell separates the gr main effect and
            the kw&times;gr interaction out of a linear ladder. Evaluation uses <M t="k=10" /> and
            server-side default parameters, with the 149 version-free NL queries all taking the
            global path. Paired significance uses the McNemar two-sided exact test, reporting the
            net flipped queries (net). The noise floor is roughly ±3 queries, coming from
            embedding-API jitter and a small amount of corpus growth.
          </>
        }
        zh={
          <>
            <Lead>消融设计。</Lead>五个 setup 构成 kw&times;gr 的
            2&times;2，再加一个外部锚点：base（iv/sv/ft）、+kw、+gr（kw
            关）、+kw+gr（全流水线），以及 Zendesk 原生搜索。补上 off-diagonal 的 +gr 一格，是为了把
            gr 主效应与 kw&times;gr 交互从线性阶梯中拆开。评测取 <M t="k=10" />
            、服务端默认参数，149 条无版本 NL 查询全部走全局路径。配对显著性用 McNemar two-sided
            exact 检验，报告净翻正查询数 net。噪声地板约为 ±3 条查询，来自 embedding API
            抖动与语料的微量增长。
          </>
        }
      />
    </Section>
  )
}
