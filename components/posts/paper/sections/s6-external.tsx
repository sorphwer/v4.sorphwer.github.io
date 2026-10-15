import type { ReactNode } from 'react'
import { C, Cite, Fig, Lead, M, P, T, TableCaption, TableShell, Td, Th } from '../shared'
import { FigExt, FigExtCost, FigScale } from '../figures'

/** Green-bold marker for the per-column winner among quality metrics. */
function Win({ children }: { children: ReactNode }) {
  return <span className="font-semibold text-ds-green-900">{children}</span>
}

export default function S6External() {
  return (
    <section id="s6-ext" className="mt-12 scroll-mt-24">
      <h3 className="text-lg font-semibold leading-7 tracking-[-0.4px] text-ds-gray-1000">
        <T en="Comparison with General-Purpose RAG Baselines" zh="与通用 RAG 基线的对比" />
      </h3>
      <div className="mt-1 font-geist-mono text-[12.5px] text-ds-gray-700">
        <T en="2026-07-08; 600-ticket rerun 2026-07-15" zh="2026-07-08；600 工单复跑 2026-07-15" />
      </div>

      <div className="mt-4">
        <P
          en={
            <>
              The ablation focuses on whether each channel works; the value of the vertical design
              is judged by comparison against general-purpose approaches: if an off-the-shelf
              general RAG ties or beats this system on the same evaluation set, domain customization
              loses its point. To that end this paper adds two external baselines on the same golden
              evaluation (§5): <strong className="font-semibold">LightRAG hybrid</strong>{' '}
              <Cite n={10} /> (lightrag-hku 1.5.4, sharing this system&rsquo;s LLM and embedding
              stack: <C>gemini-3.5-flash</C> + <C>gemini-embedding-2-preview</C>) and a{' '}
              <strong className="font-semibold">naive vector baseline</strong> (raw-text chunking +
              the same embedding model + cosine similarity, with a ticket scored by its max chunk
              score). GraphRAG <Cite n={5} /> is excluded: its global-sensemaking positioning does
              not match ticket retrieval, and its community-summary index is too costly.
            </>
          }
          zh={
            <>
              消融聚焦&ldquo;各通道是否有效&rdquo;；垂直方案的价值通过与通用方案对比评估：如果开箱即用的通用
              RAG 在同一评测集上打平或超过本系统，领域定制就失去意义。为此本文在同一套 golden
              评测（§5）上加入两个外部基线：
              <strong className="font-semibold">LightRAG hybrid</strong> <Cite n={10} />
              （lightrag-hku 1.5.4，LLM 与 embedding 与本系统同栈：<C>gemini-3.5-flash</C> +{' '}
              <C>gemini-embedding-2-preview</C>）与{' '}
              <strong className="font-semibold">naive 向量检索</strong>（原始文本分块 + 同 embedding
              模型 + 余弦相似度，工单分取 max chunk 分）。GraphRAG <Cite n={5} /> 被排除：其全局
              sensemaking 定位与工单检索不匹配，且社区摘要索引成本过高。
            </>
          }
        />
      </div>

      <P
        en={
          <>
            <Lead>Comparison protocol.</Lead> The comparison happens at the retrieval layer: the
            context returned by a baseline (chunks / entities / relations) is mapped back to ticket
            IDs via source documents and scored against <C>expected_ticket_ids</C> with the same
            metric set, introducing no LLM judge. The corpus is the 300-ticket manifest (all 234
            expected tickets of the 149 golden queries + 66 noise tickets drawn with a fixed seed).
            The baselines index the <strong className="font-semibold">raw ticket markdown</strong>{' '}
            (subject + description + full conversation) rather than this system&rsquo;s LLM
            summaries &mdash; summary extraction is part of the system under test and is not fed to
            the opponents; cost accounting, however, counts the full pipeline on both sides. This
            system runs two rows: <C>kg-full</C> (the production full corpus of 2,521 tickets, the
            real serving posture, with roughly 8&times; the distractors of the baselines) and{' '}
            <C>kg-subset</C> (results filtered to the 300-corpus for a strictly same-corpus row).
            The comparison then walks up the corpus scale &mdash; 300, 600, 2,521 tickets &mdash;
            ending with the measured cost accounting.
          </>
        }
        zh={
          <>
            <Lead>对比协议。</Lead>对比发生在检索层：基线返回的 context（chunk / 实体 /
            关系）经来源文档映射回 ticket ID，与 <C>expected_ticket_ids</C> 用同一组指标计分，不引入
            LLM judge。语料为 300 工单 manifest（149 条 golden query 的全部 234 个 expected 工单 +
            66 个固定种子采样的噪声工单）。基线索引
            <strong className="font-semibold">原始工单 markdown</strong>（subject + 描述 +
            完整对话）而非本系统的 LLM
            摘要&mdash;&mdash;摘要抽取是被测系统的一部分，不喂给对手；成本核算则两侧都计入完整
            pipeline。本系统跑两行：<C>kg-full</C>（生产全库 2521
            工单，真实服务姿势，干扰项约为基线的 8 倍）与 <C>kg-subset</C>（结果过滤到 300
            语料内的严格同语料行）。对比沿语料规模逐级展开&mdash;&mdash;300、600、2521
            工单&mdash;&mdash;最后以实测成本核算收尾。
          </>
        }
      />

      {/* Block 1 — 300-ticket same corpus */}
      <div className="mt-5">
        <TableCaption
          num={7}
          en={
            <>
              Quality on the 300-ticket same corpus (149 golden queries, <M t="k=10" />, zero
              errors; run 2026-07-08). All backends indexed the identical corpus. Green bold marks
              the per-column best quality value; the blue row is this system.
            </>
          }
          zh={
            <>
              300 工单同语料质量对比（149 条 golden query，
              <M t="k=10" />
              ，零错误；2026-07-08
              实测）。全部后端索引同一语料。绿色加粗为该列质量指标最优值；蓝色行为本系统。
            </>
          }
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="system" zh="系统" />
              </Th>
              <Th align="right">Hit@1</Th>
              <Th align="right">Hit@5</Th>
              <Th align="right">Hit@10</Th>
              <Th align="right">MRR@10</Th>
              <Th align="right">nDCG@10</Th>
              <Th align="right">p50</Th>
              <Th align="right">p95</Th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-ds-blue-100">
              <Td>
                <T en="this system (kg-subset)" zh="本系统（kg-subset）" />
              </Td>
              <Td align="right" mono>
                <Win>0.772</Win>
              </Td>
              <Td align="right" mono>
                0.966
              </Td>
              <Td align="right" mono>
                0.966
              </Td>
              <Td align="right" mono>
                <Win>0.856</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.857</Win>
              </Td>
              <Td align="right" mono>
                1.3s
              </Td>
              <Td align="right" mono>
                2.1s
              </Td>
            </tr>
            <tr>
              <Td>
                <T en="naive vector" zh="naive 向量" />
              </Td>
              <Td align="right" mono>
                0.738
              </Td>
              <Td align="right" mono>
                <Win>0.987</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.993</Win>
              </Td>
              <Td align="right" mono>
                0.841
              </Td>
              <Td align="right" mono>
                0.835
              </Td>
              <Td align="right" mono>
                1.2s
              </Td>
              <Td align="right" mono>
                2.3s
              </Td>
            </tr>
            <tr>
              <Td last>
                <T en="LightRAG hybrid" zh="LightRAG hybrid" />
              </Td>
              <Td last align="right" mono>
                0.685
              </Td>
              <Td last align="right" mono>
                0.919
              </Td>
              <Td last align="right" mono>
                0.960
              </Td>
              <Td last align="right" mono>
                0.764
              </Td>
              <Td last align="right" mono>
                0.706
              </Td>
              <Td last align="right" mono>
                6.7s
              </Td>
              <Td last align="right" mono>
                9.4s
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      <Fig
        num={13}
        en={
          <>
            Same-corpus (300 tickets) quality comparison across the three systems. This system leads
            on Hit@1 / MRR@10 / nDCG@10; the naive vector baseline&rsquo;s lead on Hit@5/10 is
            small-corpus recall saturation (see the caveat in the text). LightRAG trails on every
            metric.
          </>
        }
        zh={
          <>
            同语料（300 工单）三系统质量对比。本系统在 Hit@1 / MRR@10 / nDCG@10 全部领先；naive
            向量在 Hit@5/10 上的领先属小语料 recall 饱和（见正文 caveat）。LightRAG
            在全部指标上垫底。
          </>
        }
      >
        <FigExt />
      </Fig>

      <P
        en={
          <>
            <Lead>
              Conclusion 4: on the same corpus, this system beats LightRAG across the board at an
              order of magnitude lower cost.
            </Lead>{' '}
            Hit@1 +8.7pp (0.772 vs 0.685), MRR@10 +0.092, nDCG@10 +0.151; its indexing LLM tokens
            are 1/11.6 of LightRAG&rsquo;s (Table 11) and its median query latency 1/5 &mdash;
            LightRAG must run LLM keyword extraction before every query (the main source of its 6.7s
            p50), whereas this system&rsquo;s query path makes zero LLM calls (only the query
            embedding). This answers the justification question for the vertical design:
            domain-customized summaries + a star-shaped schema + fusion scoring beat handing the
            same batch of raw tickets to a general graph RAG.
          </>
        }
        zh={
          <>
            <Lead>结论四：同语料下本系统全面胜过 LightRAG，且成本占优一个数量级。</Lead>
            Hit@1 +8.7pp（0.772 vs 0.685）、MRR@10 +0.092、nDCG@10 +0.151；索引 LLM token 为其
            1/11.6（表 11），查询中位延迟为其 1/5&mdash;&mdash;LightRAG 每条查询必须先经 LLM
            关键词抽取（其 p50 6.7s 的主要来源），本系统查询路径零 LLM 调用（仅查询
            embedding）。这回答了垂直方案的正当性问题：领域定制的摘要 + 星型图 +
            融合打分，优于把同一批原始工单交给通用图 RAG。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>
              Conclusion 5: on self-contained single-document corpora, a generic entity graph is a
              poor buy.
            </Lead>{' '}
            LightRAG even loses to the zero-LLM-cost naive vector baseline on every quality metric.
            The dominant shape of ticket retrieval is finding the one right ticket, whereas
            LightRAG&rsquo;s entity-relation graph shines at stitching knowledge across documents
            &mdash; when the corpus shape is mismatched with the design&rsquo;s assumptions,
            10&times; the indexing cost buys no quality. Its two-tier keyword mechanism also fails
            to deliver value, which is a cautionary reference for this paper&rsquo;s kw channel (§6,
            Conclusion 3): invest in query-side keyword normalization &mdash; the generated hint
            lexicon &mdash; rather than copying an entity-graph structure, a direction measured to
            lift Hit@3/5 and confidence inside the serving window.
          </>
        }
        zh={
          <>
            <Lead>结论五：在&ldquo;单文档自包含&rdquo;语料上，通用实体图不划算。</Lead>
            LightRAG 在全部质量指标上还输给了零 LLM 成本的 naive
            向量基线。工单检索的主要形态是&ldquo;找到那张对的工单&rdquo;，而 LightRAG
            的实体-关系图优势在跨文档知识拼接&mdash;&mdash;语料形态与方案假设错配时，10
            倍索引成本买不来质量。它的双层关键词机制也未能兑现价值，这对本文 kw 通道（§6
            结论三）是一个反向参考：投入查询侧关键词归一&mdash;&mdash;即生成 hint
            词典&mdash;&mdash;而非照搬实体图结构，该方向实测在服务窗口内提升 Hit@3/5 与置信度。
          </>
        }
      />

      {/* Block 2 — 600-ticket nested rerun */}
      <div className="mt-5">
        <TableCaption
          num={8}
          en={
            <>
              Quality on the nested 600-ticket corpus (rerun 2026-07-15): the 300-ticket manifest
              plus seed-fixed noise padding up to 600 tickets &mdash; the same construction as the
              decay curve, so rows are directly comparable with Table 7. &dagger;: the LightRAG
              latency here is not comparable to its 300-ticket rows (LLM response cache, see the
              caveat).
            </>
          }
          zh={
            <>
              600 工单嵌套语料质量对比（2026-07-15 复跑）：以 300 工单 manifest 为基底、固定 seed
              补噪声至 600 工单&mdash;&mdash;与衰减曲线同构造，各行与表 7
              直接可比。&dagger;：LightRAG 此处延迟不可与其 300 行比较（LLM 响应缓存，见 caveat）。
            </>
          }
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="system" zh="系统" />
              </Th>
              <Th align="right">Hit@1</Th>
              <Th align="right">Hit@5</Th>
              <Th align="right">Hit@10</Th>
              <Th align="right">MRR@10</Th>
              <Th align="right">nDCG@10</Th>
              <Th align="right">p50</Th>
              <Th align="right">p95</Th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-ds-blue-100">
              <Td>
                <T en="this system (kg-subset)" zh="本系统（kg-subset）" />
              </Td>
              <Td align="right" mono>
                <Win>0.711</Win>
              </Td>
              <Td align="right" mono>
                0.940
              </Td>
              <Td align="right" mono>
                0.966
              </Td>
              <Td align="right" mono>
                <Win>0.819</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.816</Win>
              </Td>
              <Td align="right" mono>
                0.8s
              </Td>
              <Td align="right" mono>
                1.0s
              </Td>
            </tr>
            <tr>
              <Td>
                <T en="naive vector" zh="naive 向量" />
              </Td>
              <Td align="right" mono>
                0.604
              </Td>
              <Td align="right" mono>
                <Win>0.946</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.987</Win>
              </Td>
              <Td align="right" mono>
                0.751
              </Td>
              <Td align="right" mono>
                0.765
              </Td>
              <Td align="right" mono>
                0.7s
              </Td>
              <Td align="right" mono>
                0.9s
              </Td>
            </tr>
            <tr>
              <Td last>
                <T en="LightRAG hybrid" zh="LightRAG hybrid" />
              </Td>
              <Td last align="right" mono>
                0.570
              </Td>
              <Td last align="right" mono>
                0.872
              </Td>
              <Td last align="right" mono>
                0.960
              </Td>
              <Td last align="right" mono>
                0.684
              </Td>
              <Td last align="right" mono>
                0.644
              </Td>
              <Td last align="right" mono>
                1.7s<sup>&dagger;</sup>
              </Td>
              <Td last align="right" mono>
                3.0s<sup>&dagger;</sup>
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      <P
        en={
          <>
            <Lead>
              Conclusion 6: the 600-ticket rerun (2026-07-15) &mdash; the ordering holds with scale,
              and LightRAG gains a second measured decay point.
            </Lead>{' '}
            On a 600-ticket nested corpus built the same way as the decay curve (the 300 manifest as
            the base + seed-42 noise; report{' '}
            <C>benchmarks/retriever/reports/2026-07-15-external-600/</C>), naive rebuilt its index
            and LightRAG was incrementally expanded from the 300 index before a full rerun. The
            measured slope of a single 300&rarr;600 doubling: this system −6.1pp (0.772&rarr;0.711),
            LightRAG −11.5pp (0.685&rarr;0.570), naive −13.4pp (0.738&rarr;0.604) &mdash; the
            ordering is unchanged, and this system&rsquo;s Hit@1 lead widens with the corpus (over
            naive from +3.4pp to +10.7pp, over LightRAG from +8.7pp to +14.1pp). Two methodological
            gains: first, naive&rsquo;s <em>rebuilt index</em> (re-embedded) 0.604 differs by only 1
            query from the masked re-evaluation&rsquo;s 0.611 (seed 42, the max end of the 600 row
            in Table 10), giving end-to-end empirical support for the &ldquo;masking =
            rebuild&rdquo; equivalence of Conclusion 8 below; second, the LightRAG index can be
            incrementally expanded from the 300 base (doc-status skips already-processed documents),
            and +300 tickets measured about 5.1M LLM tokens and 12.7M embedding tokens (the
            entity/relation re-embedding during merging being the bulk), cumulatively 13.0M / 14.2M
            &mdash; incremental expansion does not dilute its cost disadvantage.
          </>
        }
        zh={
          <>
            <Lead>
              结论六：600 工单复跑（2026-07-15）&mdash;&mdash;排序不随语料翻转，并为 LightRAG
              补上第二个实测衰减点。
            </Lead>
            在与衰减曲线同构造的 600 工单嵌套语料上（300 manifest 为基底 + seed-42 补噪声，报告{' '}
            <C>benchmarks/retriever/reports/2026-07-15-external-600/</C>），naive 重建索引、LightRAG
            由 300 索引增量扩展后全量复跑。300&rarr;600 一次翻倍的实测斜率：本系统
            −6.1pp（0.772&rarr;0.711）、LightRAG −11.5pp（0.685&rarr;0.570）、naive
            −13.4pp（0.738&rarr;0.604）&mdash;&mdash;排序保持不变，本系统的 Hit@1 领先随语料扩大（对
            naive 从 +3.4pp 扩至 +10.7pp，对 LightRAG 从 +8.7pp 扩至
            +14.1pp）。两项方法学收获：其一，naive <em>重建索引</em>（重新 embedding）的 0.604
            与掩码复评的 0.611（seed 42，表 10 600 行的 max 端）仅差 1
            条查询，为下文结论八的&ldquo;掩码 = 重建&rdquo;等价性提供了端到端实证；其二，LightRAG
            索引可由 300 基底增量扩展（doc-status 跳过已处理文档），+300 工单实测花费约 5.1M LLM
            tokens 与 12.7M embedding tokens（合并期实体/关系重嵌入为大头），累计 13.0M /
            14.2M&mdash;&mdash;增量扩展并未摊薄其成本劣势。
          </>
        }
      />

      {/* Block 3 — production scale, 2,521 tickets */}
      <div className="mt-5">
        <TableCaption
          num={9}
          en={
            <>
              Quality on the production-scale corpus, 2,521 tickets (same 149 queries; naive
              full-corpus rerun 2026-07-09). <C>kg-full</C> is the real serving posture. The
              LightRAG full-corpus rerun was not done (~65M extraction tokens); only backends
              actually re-indexed at this scale are shown.
            </>
          }
          zh={
            <>
              生产规模语料质量对比，2521 工单（同一 149 条查询；naive 全库复跑 2026-07-09）。
              <C>kg-full</C> 为真实服务姿势。LightRAG 全库复跑未做（抽取预算约 65M
              token）；仅展示实际在该规模重建过索引的后端。
            </>
          }
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="system" zh="系统" />
              </Th>
              <Th align="right">Hit@1</Th>
              <Th align="right">Hit@5</Th>
              <Th align="right">Hit@10</Th>
              <Th align="right">MRR@10</Th>
              <Th align="right">nDCG@10</Th>
              <Th align="right">p50</Th>
              <Th align="right">p95</Th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-ds-blue-100">
              <Td>
                <T en="this system (kg-full, full corpus)" zh="本系统（kg-full，全库）" />
              </Td>
              <Td align="right" mono>
                <Win>0.490</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.866</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.946</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.658</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.685</Win>
              </Td>
              <Td align="right" mono>
                0.8s
              </Td>
              <Td align="right" mono>
                1.0s
              </Td>
            </tr>
            <tr>
              <Td last>
                <T
                  en="naive vector (full-corpus rerun, 2026-07-09)"
                  zh="naive 向量（全库复跑，2026-07-09）"
                />
              </Td>
              <Td last align="right" mono>
                0.242
              </Td>
              <Td last align="right" mono>
                0.752
              </Td>
              <Td last align="right" mono>
                0.906
              </Td>
              <Td last align="right" mono>
                0.454
              </Td>
              <Td last align="right" mono>
                0.499
              </Td>
              <Td last align="right" mono>
                0.6s
              </Td>
              <Td last align="right" mono>
                &mdash;
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      <P
        en={
          <>
            <Lead>
              Conclusion 7: naive&rsquo;s lead is a small-corpus artifact; at production scale the
              vertical design wins across the board.
            </Lead>{' '}
            The naive baseline&rsquo;s Hit@5/10 lead on the 300 corpus (0.987/0.993) looks like
            recall saturation, and the full-corpus rerun (2,521 tickets, 2026-07-09) confirms it: as
            distractors grow from 66 to 2,287, naive&rsquo;s Hit@1 falls from 0.738 to 0.242
            (−49.6pp), decaying about twice as fast as this system (kg over the same period
            0.772&rarr;0.490, −28.2pp). On the same 2,521 corpus, this system&rsquo;s advantage over
            naive is Hit@1 +24.8pp, MRR@10 +0.204, nDCG@10 +0.186, leading on every metric. Pure
            vectors&rsquo; verbatim fuzzy matching quickly loses discriminative power when
            near-neighbor tickets are dense, whereas multi-channel fusion and gr re-ranking provide
            independent evidence axes that keep the ranking stable under distractors &mdash; the
            larger the corpus, the clearer the value of vertical structure.
          </>
        }
        zh={
          <>
            <Lead>结论七：naive 的领先是小语料假象，生产规模下垂直方案全面胜出。</Lead>
            naive 在 300 语料上的 Hit@5/10 领先（0.987/0.993）疑似 recall 饱和，全库复跑（2521
            工单，2026-07-09）证实了这一点：干扰项从 66 涨到 2,287 后，naive 的 Hit@1 从 0.738 跌至
            0.242（−49.6pp），衰减速度约为本系统的两倍（kg 同期 0.772&rarr;0.490，−28.2pp）。同为
            2521 语料，本系统对 naive 的优势为 Hit@1 +24.8pp、MRR@10 +0.204、nDCG@10
            +0.186，本系统在全部指标上领先。纯向量的&ldquo;逐字模糊匹配&rdquo;在近邻工单密集时迅速失去区分度，而多通道融合与
            gr
            重排提供的独立证据轴使排序在干扰下更稳定&mdash;&mdash;规模越大，垂直结构的价值越明显。
          </>
        }
      />

      {/* Scale story — hero figure */}
      <Fig
        num={14}
        en={
          <>
            Decay of Hit@1 with corpus scale (log x-axis, the same 149 golden queries). The naive
            curve is a nested-subset re-evaluation (9 sizes × 3 sampling seeds; the gray band is the
            cross-seed min&ndash;max; the 300 and 2521 endpoints are fixed by construction and
            bit-for-bit consistent with Tables 7 and 9). The fit is{' '}
            <M t="\text{Hit@1} = 2.121 - 0.241\ln N" /> (<M t="R^2=0.997" />
            ): each corpus doubling drops about 16.7pp. This system&rsquo;s three points (the 600
            point is measured in the 2026-07-15 rerun, the dashed connector is illustrative only)
            give a 300&rarr;2521 two-point slope of about −9.2pp/doubling; LightRAG&rsquo;s two
            points are measured (300/600) with a slope of −11.5pp/doubling, between the two.
          </>
        }
        zh={
          <>
            Hit@1 随语料规模的衰减（对数横轴，同一 149 条 golden query）。naive
            曲线为嵌套子集复评（9 个规模 × 3 个采样 seed，灰带为跨 seed min&ndash;max；300 与 2521
            两端点由构造固定、与表 7、表 9 逐位一致）。拟合{' '}
            <M t="\text{Hit@1} = 2.121 - 0.241\ln N" /> (<M t="R^2=0.997" />
            )：每翻倍语料掉约 16.7pp。本系统三点（600 点为 2026-07-15
            复跑实测，虚线连接仅示意），300&rarr;2521 两点斜率约 −9.2pp/倍；LightRAG
            两点实测（300/600），斜率 −11.5pp/倍，介于两者之间。
          </>
        }
      >
        <FigScale />
      </Fig>

      <P
        en={
          <>
            <Lead>
              Conclusion 8: pure-vector decay follows a log-linear law; vertical structure nearly
              halves the slope.
            </Lead>{' '}
            To characterize the curve shape between the two endpoints, this paper added a
            decay-curve experiment (2026-07-09): naive&rsquo;s ticket score is per-ticket
            independent (max chunk cosine, unrelated to other candidates), so masking the candidates
            to any subset over the full index is <em>exactly equivalent</em> to re-evaluating after
            rebuilding the index on that subset &mdash; the whole curve at zero rebuild cost. The
            subsets are built by nesting (the 300-corpus manifest as the base, adding noise from the
            full corpus level by level with a fixed seed, the 234 expected tickets always in the
            library), 9 sizes × 3 seeds. The result (Table 10, Figure 14): naive&rsquo;s Hit@1 is
            strictly linear in <M t="\ln N" /> (
            <M t="R^2=0.997" />
            ), losing 16.7pp per corpus doubling and extrapolating to zero near ~6,600 tickets
            &mdash; in practice it bottoms out at the share of &ldquo;queries containing a unique
            lexical anchor (a unique term in the query that can be matched exactly)&rdquo;, but the
            trend is clear:{' '}
            <em>
              pure-vector retrieval&rsquo;s Hit@1 slides log-linearly toward unusable as the corpus
              grows
            </em>
            . This system&rsquo;s two-point slope is −9.2pp/doubling, about 55% of naive&rsquo;s;
            read another way, naive&rsquo;s precision at about 900 tickets (0.501) has already
            fallen to this system&rsquo;s level at 2,521 tickets (0.490) &mdash; the vertical
            structure supports about 2.8&times; the corpus scale. Mechanistically, the decay comes
            from extreme-value noise: the gold ticket must beat the maximum similarity of all{' '}
            <M t="N-1" /> distractors, and that maximum grows slowly with <M t="N" />, continually
            compressing the gold ticket&rsquo;s margin; this system&rsquo;s slope advantage comes
            from the wider margin of summary embeddings, the consensus dilution of single-channel
            extreme-value noise by multi-channel RRF, and the hedging effect of gr co-citation
            evidence growing denser as the corpus grows &mdash; the last being the only signal
            source that strengthens with <M t="N" />.
          </>
        }
        zh={
          <>
            <Lead>结论八：纯向量的衰减是 log-linear 规律，垂直结构把斜率近乎减半。</Lead>
            为刻画两端点之间的曲线形状，本文补做了衰减曲线实验（2026-07-09）：naive
            的工单分是逐票独立的（max chunk
            余弦，与其它候选无关），因此在全量索引上把候选掩码到任意子集，
            <em>精确等价于</em>
            在该子集上重建索引后评测&mdash;&mdash;整条曲线零重建成本。子集按嵌套构造（300 语料
            manifest 为基底，逐级用固定 seed 从全库补噪声，234 个 expected 工单恒在库），9 个规模 ×
            3 个 seed。结果（表 10、图 14）：naive 的 Hit@1 对 <M t="\ln N" /> 严格线性（
            <M t="R^2=0.997" />
            ），每翻倍语料损失 16.7pp，外推至约 6,600
            工单时归零&mdash;&mdash;实际会在&ldquo;查询含唯一词法锚点（查询中可精确匹配的唯一词）&rdquo;的占比处触底，但趋势明确：
            <em>纯向量检索的 Hit@1 随语料规模对数线性地滑向不可用</em>。本系统两点斜率为
            −9.2pp/倍，约为 naive 的 55%；换一种读法，naive 在约 900
            工单时的精度（0.501）已跌到本系统在 2,521
            工单下的水平（0.490）&mdash;&mdash;垂直结构支撑了约 2.8
            倍的语料规模。机制上，衰减来自极值噪声：金票必须打赢全部 <M t="N-1" />{' '}
            个干扰项的最大相似度，该最大值随 <M t="N" />{' '}
            缓慢增长，持续压缩金票边际；本系统的斜率优势来自摘要嵌入的更宽边际、多通道 RRF
            对单通道极值噪声的共识稀释，以及 gr
            共引证据随语料增长而变密的对冲效应&mdash;&mdash;后者是唯一随 <M t="N" /> 增强的信号源。
          </>
        }
      />

      <div className="mt-5">
        <TableCaption
          num={10}
          en={
            <>
              Naive vector Hit@1 decay curve (nested subsets, 3 seeds, means and cross-seed ranges;
              the two endpoints are seed-invariant). Full metrics in{' '}
              <C>benchmarks/retriever/reports/2026-07-09-naive-decay/</C>.
            </>
          }
          zh={
            <>
              naive 向量 Hit@1 衰减曲线（嵌套子集，3 seeds，均值与跨 seed 极差；两端点 seed
              不变量）。完整指标见 <C>benchmarks/retriever/reports/2026-07-09-naive-decay/</C>。
            </>
          }
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="corpus" zh="语料" />
              </Th>
              <Th align="right">Hit@1</Th>
              <Th align="right">Hit@5</Th>
              <Th align="right">MRR@10</Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Td mono>300</Td>
              <Td align="right" mono>
                0.738
              </Td>
              <Td align="right" mono>
                0.987
              </Td>
              <Td align="right" mono>
                0.841
              </Td>
            </tr>
            <tr>
              <Td mono>450</Td>
              <Td align="right" mono>
                <T en="0.651 (0.638–0.658)" zh="0.651（0.638–0.658）" />
              </Td>
              <Td align="right" mono>
                0.971
              </Td>
              <Td align="right" mono>
                0.786
              </Td>
            </tr>
            <tr>
              <Td mono>600</Td>
              <Td align="right" mono>
                <T en="0.579 (0.544–0.611)" zh="0.579（0.544–0.611）" />
              </Td>
              <Td align="right" mono>
                0.946
              </Td>
              <Td align="right" mono>
                0.737
              </Td>
            </tr>
            <tr>
              <Td mono>900</Td>
              <Td align="right" mono>
                <T en="0.501 (0.477–0.537)" zh="0.501（0.477–0.537）" />
              </Td>
              <Td align="right" mono>
                0.911
              </Td>
              <Td align="right" mono>
                0.676
              </Td>
            </tr>
            <tr>
              <Td mono>1200</Td>
              <Td align="right" mono>
                <T en="0.414 (0.403–0.436)" zh="0.414（0.403–0.436）" />
              </Td>
              <Td align="right" mono>
                0.888
              </Td>
              <Td align="right" mono>
                0.616
              </Td>
            </tr>
            <tr>
              <Td mono>1600</Td>
              <Td align="right" mono>
                <T en="0.327 (0.315–0.342)" zh="0.327（0.315–0.342）" />
              </Td>
              <Td align="right" mono>
                0.843
              </Td>
              <Td align="right" mono>
                0.548
              </Td>
            </tr>
            <tr>
              <Td mono>2000</Td>
              <Td align="right" mono>
                <T en="0.277 (0.268–0.289)" zh="0.277（0.268–0.289）" />
              </Td>
              <Td align="right" mono>
                0.796
              </Td>
              <Td align="right" mono>
                0.500
              </Td>
            </tr>
            <tr>
              <Td mono>2250</Td>
              <Td align="right" mono>
                <T en="0.262 (0.255–0.268)" zh="0.262（0.255–0.268）" />
              </Td>
              <Td align="right" mono>
                0.774
              </Td>
              <Td align="right" mono>
                0.479
              </Td>
            </tr>
            <tr>
              <Td last mono>
                2521
              </Td>
              <Td last align="right" mono>
                0.242
              </Td>
              <Td last align="right" mono>
                0.752
              </Td>
              <Td last align="right" mono>
                0.454
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      {/* Block 4 — indexing & query cost */}
      <div className="mt-5">
        <TableCaption
          num={11}
          en={
            <>
              Indexing and query cost (measured on 300 tickets; tokens estimated as chars/4).
              Per-ticket cost extrapolates linearly with corpus scale.
            </>
          }
          zh={
            <>
              索引与查询成本（300 工单实测；token 为 chars/4
              估算）。每工单成本可按语料规模线性外推。
            </>
          }
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="system" zh="系统" />
              </Th>
              <Th align="right">
                <T en="LLM calls" zh="LLM 调用" />
              </Th>
              <Th align="right">
                <T en="LLM tokens" zh="LLM tokens" />
              </Th>
              <Th align="right">
                <T en="embedding vectors" zh="Embedding 向量" />
              </Th>
              <Th align="right">
                <T en="embedding tokens" zh="Embedding tokens" />
              </Th>
              <Th align="right">
                <T en="query-time LLM calls" zh="查询时 LLM 调用" />
              </Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Td>
                <T en="this system ingest pipeline" zh="本系统 ingest pipeline" />
              </Td>
              <Td align="right" mono>
                <T en="300 (1/ticket)" zh="300（1/工单）" />
              </Td>
              <Td align="right" mono>
                680K
              </Td>
              <Td align="right" mono>
                <T en="600 (2/ticket)" zh="600（2/工单）" />
              </Td>
              <Td align="right" mono>
                128K
              </Td>
              <Td align="right" mono strong>
                0
              </Td>
            </tr>
            <tr>
              <Td>
                <T en="naive vector" zh="naive 向量" />
              </Td>
              <Td align="right" mono>
                0
              </Td>
              <Td align="right" mono>
                0
              </Td>
              <Td align="right" mono>
                <T en="799 (~2.7/ticket)" zh="799（~2.7/工单）" />
              </Td>
              <Td align="right" mono>
                574K
              </Td>
              <Td align="right" mono>
                0
              </Td>
            </tr>
            <tr>
              <Td last>
                <T en="LightRAG hybrid" zh="LightRAG hybrid" />
              </Td>
              <Td last align="right" mono>
                <T en="2,269 (~7.6/ticket)" zh="2,269（~7.6/工单）" />
              </Td>
              <Td last align="right" mono>
                <span className="font-semibold text-ds-red-900">
                  <T en="7.87M (11.6×)" zh="7.87M（11.6×）" />
                </span>
              </Td>
              <Td last align="right" mono>
                <T en="18,032 (~60/ticket)" zh="18,032（~60/工单）" />
              </Td>
              <Td last align="right" mono>
                <T en="1.39M (10.9×)" zh="1.39M（10.9×）" />
              </Td>
              <Td last align="right" mono>
                <T en="≥1 (keyword extraction)" zh="≥1（关键词抽取）" />
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      <Fig
        num={15}
        en={
          <>
            Total indexing tokens compared (LLM + embedding, log y-axis): LightRAG is roughly
            11&times; this system. Its cost structure comes from chunking at ~1.2K tokens and
            carrying the extraction prompt per chunk, gleaning re-queries, and vectorizing each
            entity and relation separately (~60 vectors/ticket vs this system&rsquo;s 2
            vectors/ticket).
          </>
        }
        zh={
          <>
            索引 token 总量对比（LLM + embedding，对数纵轴）：LightRAG 为本系统的约 11
            倍。其成本结构来自按 ~1.2K token 分块后逐块携带抽取 prompt、gleaning
            复问，以及每个实体与关系单独向量化（~60 向量/工单 vs 本系统 2 向量/工单）。
          </>
        }
      >
        <FigExtCost />
      </Fig>

      <P
        en={
          <>
            <Lead>Caveat.</Lead> kg-subset has an underfill after top-20 filtering (Hit@5 = Hit@10 =
            0.966), so its true Hit@10 is underestimated. The mid-scale points of this
            system&rsquo;s decay curve require independent Neo4j sub-databases (channel scoring and
            df statistics depend on the full in-library corpus, so masked re-evaluation is inexact
            for kg), and −9.2pp/doubling is the 300&rarr;2521 two-point slope, not a fit;
            naive&rsquo;s masked re-evaluation, by contrast, is exact (per-ticket independent
            scoring). <Lead>&dagger;</Lead>: the query latency of the LightRAG 600 rows is not
            comparable to its 300 rows &mdash; the 600 index was copy-expanded from the 300 working
            directory, and 145 of the 149 queries hit the keyword-extraction LLM response cache
            copied along with it, so its 1.7s p50 largely excludes LLM round-trips; the 6.7s of the
            300 rows is the real cold-cache query cost. LightRAG&rsquo;s full-corpus rerun (indexing
            budget about 65M tokens) is not yet complete, but the 600 rerun has already given its
            second measured point: the decay slope is between this system and naive, and its ranking
            metrics still trail across the board.
          </>
        }
        zh={
          <>
            <Lead>Caveat：</Lead>kg-subset 存在 top-20 过滤后的欠填充（Hit@5=Hit@10=0.966），其真实
            Hit@10 被低估。本系统的衰减曲线中间规模需要独立的 Neo4j 子库（通道打分与 df
            统计依赖库内全量，掩码复评对 kg 不精确），−9.2pp/倍是 300&rarr;2521
            两点斜率、非拟合；naive 的掩码复评则是精确的（逐票独立打分）。<Lead>&dagger;</Lead>
            ：LightRAG 600 行的查询延迟不可与其 300 行比较&mdash;&mdash;600 索引自 300
            工作目录复制扩展，149 条查询中 145 条命中了随之复制的关键词抽取 LLM 响应缓存，其 p50
            1.7s 基本不含 LLM 往返；300 行的 6.7s 才代表冷缓存的真实查询成本。LightRAG
            的全量复跑（索引预算约 65M tokens）仍未完成，但 600
            复跑已给出其第二个实测点：衰减斜率介于本系统与 naive 之间，排名指标仍全面垫底。
          </>
        }
      />
    </section>
  )
}
