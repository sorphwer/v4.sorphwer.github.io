import type { ReactNode } from 'react'
import { C, Fig, Lead, M, P, Section, T, TableCaption, TableShell, Td, Th } from '../shared'
import { FigFinger, FigLatency, FigMain, FigMcnemar } from '../figures'

/** Green-bold marker for the per-column best value among KG paths. */
function Win({ children }: { children: ReactNode }) {
  return <span className="font-semibold text-ds-green-900">{children}</span>
}

export default function S6Results() {
  return (
    <Section id="s6" num="06" en="Results & Analysis" zh="评测结果与分析">
      {/* Table 5 — Golden Query main results (§1 links to #tbl-main) */}
      <div id="tbl-main" className="mt-6 scroll-mt-24">
        <TableCaption
          num={5}
          en={
            <>
              Golden Query main results (149 frozen queries, <M t="k" />
              &le;10, run 2026-07-09, corpus N=2,564). kw runs the generated hint lexicon &mdash;
              the production posture since 2026-07-15. mean max-conf = the mean of the per-query
              maximum confidence. Green bold marks the per-column best among KG paths; the blue row
              is the production configuration. The Zendesk anchor searches its own index
              (corpus-independent, measured 2026-07-07); per-setup latency is reported in Figure 12.
            </>
          }
          zh={
            <>
              Golden Query 主结果（149 条冻结查询，
              <M t="k" />
              ≤10，2026-07-09 实测，语料 N=2,564）。kw 使用生成词典&mdash;&mdash;即 2026-07-15
              起的生产姿势。mean max-conf = 每查询最大 confidence 的均值。绿色加粗为 KG
              路径中该列最优；蓝色行为生产配置。Zendesk 锚点搜索其自有索引（与本语料无关，2026-07-07
              实测）；各档延迟见图 12。
            </>
          }
        />
        <TableShell>
          <thead>
            <tr>
              <Th>setup</Th>
              <Th align="right">Hit@1</Th>
              <Th align="right">Hit@3</Th>
              <Th align="right">Hit@5</Th>
              <Th align="right">Hit@10</Th>
              <Th align="right">MRR@10</Th>
              <Th align="right">nDCG@10</Th>
              <Th align="right">max-conf</Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Td>
                <T en="base (iv/sv/ft)" zh="base（iv/sv/ft）" />
              </Td>
              <Td align="right" mono>
                0.362
              </Td>
              <Td align="right" mono>
                0.718
              </Td>
              <Td align="right" mono>
                0.872
              </Td>
              <Td align="right" mono>
                <Win>0.953</Win>
              </Td>
              <Td align="right" mono>
                0.577
              </Td>
              <Td align="right" mono>
                0.634
              </Td>
              <Td align="right" mono>
                0.685
              </Td>
            </tr>
            <tr>
              <Td>+kw</Td>
              <Td align="right" mono>
                0.362
              </Td>
              <Td align="right" mono>
                0.785
              </Td>
              <Td align="right" mono>
                0.886
              </Td>
              <Td align="right" mono>
                <Win>0.953</Win>
              </Td>
              <Td align="right" mono>
                0.589
              </Td>
              <Td align="right" mono>
                0.644
              </Td>
              <Td align="right" mono>
                <Win>0.709</Win>
              </Td>
            </tr>
            <tr>
              <Td>
                <T en="+gr (kw off)" zh="+gr（kw 关）" />
              </Td>
              <Td align="right" mono>
                <Win>0.463</Win>
              </Td>
              <Td align="right" mono>
                0.785
              </Td>
              <Td align="right" mono>
                0.859
              </Td>
              <Td align="right" mono>
                0.946
              </Td>
              <Td align="right" mono>
                <Win>0.638</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.671</Win>
              </Td>
              <Td align="right" mono>
                0.675
              </Td>
            </tr>
            <tr className="bg-ds-blue-100">
              <Td>
                <T en="+kw+gr (full pipeline, production)" zh="+kw+gr（全流水线，生产配置）" />
              </Td>
              <Td align="right" mono>
                0.450
              </Td>
              <Td align="right" mono>
                <Win>0.826</Win>
              </Td>
              <Td align="right" mono>
                <Win>0.906</Win>
              </Td>
              <Td align="right" mono>
                0.933
              </Td>
              <Td align="right" mono>
                <Win>0.638</Win>
              </Td>
              <Td align="right" mono>
                0.662
              </Td>
              <Td align="right" mono>
                <Win>0.709</Win>
              </Td>
            </tr>
            <tr>
              <Td last muted>
                <T en="zendesk native search" zh="zendesk 原生搜索" />
              </Td>
              <Td last align="right" mono muted>
                0.000
              </Td>
              <Td last align="right" mono muted>
                0.013
              </Td>
              <Td last align="right" mono muted>
                0.027
              </Td>
              <Td last align="right" mono muted>
                0.027
              </Td>
              <Td last align="right" mono muted>
                0.009
              </Td>
              <Td last align="right" mono muted>
                0.010
              </Td>
              <Td last align="right" mono muted>
                &mdash;
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      <Fig
        num={9}
        en={
          <>
            Comparison of the four KG setups plus the Zendesk-native external anchor on Hit@1 /
            Hit@5 / MRR@10 / nDCG@10. gr opens the gap on Hit@1 and the ranking-quality metrics,
            while kw adds its gain inside the serving window (Hit@5, and Hit@3 in Table 5); the red
            Zendesk bars sit at the zero line (Hit@1 0.000, Hit@5 0.027), an order-of-magnitude gap
            to any KG path.
          </>
        }
        zh={
          <>
            四档 KG setup 加 Zendesk 原生搜索外部锚点在 Hit@1 / Hit@5 / MRR@10 / nDCG@10
            上的对比。gr 在 Hit@1 与排序质量指标上拉开差距，kw 的增益体现在服务窗口内（Hit@5，及表 5
            的 Hit@3）；红色 Zendesk 柱贴在零线（Hit@1 0.000、Hit@5 0.027），与任一 KG
            路径都差出量级。
          </>
        }
      >
        <FigMain />
      </Fig>

      <Fig
        num={10}
        en={
          <>
            gr&rsquo;s pure re-ranking fingerprint (base vs +gr): Hit@1 rises markedly
            (0.362&rarr;0.463) while Hit@5 (0.872&rarr;0.859) and Hit@10 (0.953&rarr;0.946) stay
            essentially level. This shows that gr keeps the gold-ticket recall set unchanged and
            pushes gold tickets that were &ldquo;already in the top 5 but not at rank 1&rdquo; up to
            the first position.
          </>
        }
        zh={
          <>
            gr 的纯重排指纹（base vs +gr）：Hit@1 明显上升（0.362→0.463），Hit@5 （0.872→0.859）与
            Hit@10（0.953→0.946）基本持平。这说明 gr 保持金票召回集合不变，并把&ldquo;已在前
            5、但不在第 1&rdquo;的金票推到了首位。
          </>
        }
      >
        <FigFinger />
      </Fig>

      {/* Table 6 — paired McNemar on Hit@1 */}
      <div className="mt-6">
        <TableCaption
          num={6}
          en={
            <>
              Paired McNemar on Hit@1 (two-sided exact; net = the number of queries net-flipped to
              correct by the change).
            </>
          }
          zh={<>Hit@1 上的配对 McNemar（two-sided exact；net = 该改动净翻正的查询数）。</>}
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="comparison" zh="对比" />
              </Th>
              <Th>
                <T en="meaning" zh="含义" />
              </Th>
              <Th align="right">net</Th>
              <Th align="right">p</Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Td mono>base &rarr; +kw</Td>
              <Td>
                <T en="kw main effect" zh="kw 主效应" />
              </Td>
              <Td align="right" mono>
                0
              </Td>
              <Td align="right" mono>
                1.000
              </Td>
            </tr>
            <tr>
              <Td mono>base &rarr; +gr</Td>
              <Td>
                <T en="gr main effect" zh="gr 主效应" />
              </Td>
              <Td align="right" mono strong>
                +15
              </Td>
              <Td align="right" mono strong>
                0.006
              </Td>
            </tr>
            <tr>
              <Td mono>+kw &rarr; +kw+gr</Td>
              <Td>
                <T en="gr | kw" zh="gr | kw" />
              </Td>
              <Td align="right" mono>
                +13
              </Td>
              <Td align="right" mono>
                0.002
              </Td>
            </tr>
            <tr>
              <Td mono>+gr &rarr; +kw+gr</Td>
              <Td>
                <T en="kw | gr" zh="kw | gr" />
              </Td>
              <Td align="right" mono>
                &minus;2
              </Td>
              <Td align="right" mono>
                0.815
              </Td>
            </tr>
            <tr>
              <Td last mono>
                base &rarr; +kw+gr
              </Td>
              <Td last>
                <T en="full-pipeline total gain" zh="全流水线总增益" />
              </Td>
              <Td last align="right" mono>
                +13
              </Td>
              <Td last align="right" mono>
                0.019
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      <Fig
        num={11}
        en={
          <>
            McNemar net flipped queries (bars annotated with p-values): the gr main effect +15
            (p=0.006) and gr|kw +13 (p=0.002) are significant; the kw main effect 0 (p=1.000) and
            kw|gr &minus;2 (p=0.815) fall within the noise floor &mdash; kw&rsquo;s gain lives at
            Hit@3/5, not rank 1 (Conclusion 3).
          </>
        }
        zh={
          <>
            McNemar 净翻正查询数 net（条上标 p 值）：gr 主效应 +15（p=0.006）、gr|kw
            +13（p=0.002）显著；kw 主效应 0（p=1.000）、kw|gr
            −2（p=0.815）落在噪声地板内&mdash;&mdash;kw 的增益在 Hit@3/5，不在第 1 名（结论三）。
          </>
        }
      >
        <FigMcnemar />
      </Fig>

      <Fig
        num={12}
        en={
          <>
            p50 median latency (log y-axis, ms; run 2026-07-07): all four KG paths run on the order
            of a second (726&ndash;936 ms), while Zendesk native search at 26432 ms is roughly
            30&times; higher.
          </>
        }
        zh={
          <>
            p50 中位延迟（对数纵轴，ms；2026-07-07 实测）：四档 KG
            路径均在秒级（726–936ms），Zendesk 原生搜索 26432ms 高出约 30×。
          </>
        }
      >
        <FigLatency />
      </Fig>

      <P
        en={
          <>
            <Lead>
              Conclusion 1: gr is an independent ranking-evidence axis and carries this
              round&rsquo;s main gain.
            </Lead>{' '}
            The gr main effect lifts Hit@1 from 0.362 to 0.463 (+10.1pp, net +15, p=0.006),
            statistically indistinguishable from the full pipeline&rsquo;s 0.450 (kw | gr net
            &minus;2, p=0.815). The reason is that the baseline&rsquo;s main room for improvement
            sits in within-window ordering. Between the top 5, RRF score gaps are often only 0.001
            to 0.01; gr uses an evidence axis independent of text similarity &mdash; &ldquo;which
            high-scoring tickets does this ticket share niche features with&rdquo; &mdash; to break
            these near-ties.
          </>
        }
        zh={
          <>
            <Lead>结论一：gr 是独立的排序证据轴，且承担了本轮主要增益。</Lead>gr 主效应把 Hit@1 从
            0.362 提到 0.463（+10.1pp，net +15，p=0.006），与全流水线的 0.450 在统计上不可区分（kw |
            gr net −2，p=0.815）。原因在于，基线的主要改进空间集中在窗口内定序。前 5 名之间的 RRF
            分差常常只有 0.001 到 0.01，gr
            用&ldquo;该票与哪些高分票共享小众特征&rdquo;这条独立于文本相似度的证据轴，打破了这些近似平局。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Conclusion 2: gr re-ranks; recall depth stays unchanged.</Lead> The three direct
            channels each scan the full corpus and take 10; gr&rsquo;s scope is limited to the
            seed&rsquo;s one-hop niche-feature neighborhood, examining at most 500 tickets and
            emitting 50. By P7, the contribution upper bound of gr-only newcomers, 0.0115, stays
            below the serving-window threshold; the &minus;0.7pp on Hit@10 is also consistent with
            the re-ranking account that &ldquo;a non-gold ticket promoted by gr crowds a gold ticket
            out at the tail of the window&rdquo;.
          </>
        }
        zh={
          <>
            <Lead>结论二：gr 不是更深的召回。</Lead>三条直接通道各自扫全库取 10；gr 的考察范围限定为
            seed 的一跳小众特征邻域，最多检查 500 张、输出 50 张。根据 P7，gr-only 新票的贡献上界
            0.0115 保持在服务窗口阈值以下；Hit@10 的 −0.7pp，也符合&ldquo;某张被 gr
            提拔的非金票在窗口尾部挤掉一条金票&rdquo;的重排解释。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>
              Conclusion 3: kw&rsquo;s contribution lands in the serving window and in confidence;
              rank 1 is governed by fusion consensus.
            </Lead>{' '}
            The kw channel matches query-side anchors against a hint lexicon generated from the
            graph Keyword vocabulary (rarity floor <M t="df \le \lfloor\sqrt{N+1}\rfloor-1" />, the
            same rule as gr co-citation; 8,661 entries at N=2,664; regenerated weekly by CI against
            vocabulary drift and shipped as a PR; the production posture since 2026-07-15, toggled
            via <C>RETRIEVER_HINT_LEXICON</C>). Its measurable gains sit inside the serving window:
            a Hit@3 main effect of net +10 (p=0.021), and on top of gr it lifts Hit@5
            0.859&rarr;0.906 (net +7, p=0.039 &mdash; the serving window is exactly top-5); wide
            anchor coverage also keeps confidence high (mean max-conf 0.709 for the full pipeline vs
            0.675 for gr alone). Rank 1, by contrast, does not move (kw | gr net &minus;2, p=0.815):
            the near-flat vote values <M t="1/(60+r)" /> (Figure 4) cannot overturn a rank-1
            incumbent that already holds consistent nominations from multiple channels &mdash;
            rank-1 movement is a structural property of the fusion layer, and the next jump requires
            score-aware fusion or wiring query-side anchors into gr; see{' '}
            <a href="#s8" className="text-ds-blue-900 hover:underline">
              §8
            </a>
            .
          </>
        }
        zh={
          <>
            <Lead>结论三：kw 的贡献在服务窗口与置信度；第 1 名由融合共识决定。</Lead>
            kw 通道把查询侧锚点与从图侧 Keyword 词汇表生成的 hint 词典匹配（稀有度地板{' '}
            <M t="df \le \lfloor\sqrt{N+1}\rfloor-1" />
            ，与 gr 共引同规则；N=2,664 下 8,661 词条；每周 CI 对词汇表漂移自动重生成并以 PR
            提交；即 2026-07-15 起的生产姿势，经 <C>RETRIEVER_HINT_LEXICON</C>{' '}
            切换）。其可测增益在服务窗口内：Hit@3 主效应 net +10（p=0.021），在 gr 之上把 Hit@5 从
            0.859 提至 0.906（net +7，p=0.039&mdash;&mdash;服务窗口恰为
            top-5）；宽锚点覆盖也让置信度保持高位（全流水线 mean max-conf 0.709，gr 单独为
            0.675）。第 1 名则不动（kw | gr net −2，p=0.815）：
            <M t="1/(60+r)" /> 的近平坦票值（图
            4）翻不过已有多通道一致提名的首位在位票&mdash;&mdash;rank-1
            的移动是融合层的结构性质，下一跳需要分数感知融合或把查询侧锚点接入 gr，见{' '}
            <a href="#s8" className="text-ds-blue-900 hover:underline">
              §8
            </a>
            。
          </>
        }
      />

      {/* Scope-limitation callout */}
      <div className="mt-5 rounded-md bg-ds-background-200 px-[18px] py-3.5 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
        <p className="text-sm leading-[1.85] text-ds-gray-900 text-pretty">
          <T
            en={
              <>
                <Lead>Scope.</Lead> All the conclusions above cover the global path only, because
                the Golden Queries are all version-unconstrained NL queries. This round&rsquo;s
                evaluation scope does not yet cover kw&rsquo;s in-pool re-ranking on the constrained
                path; before changing the constrained path, a version-constrained Golden Query set
                must first be built. Here one must also distinguish the bit-level invariant between
                flag-off and flag-on within the same round from the roughly &plusmn;3-query drift
                across days. The two mean different things, but both hold in the current results.
              </>
            }
            zh={
              <>
                <Lead>作用域限定：</Lead>以上结论只覆盖全局路径，因为 Golden Query
                全部是无版本约束的 NL 查询。本轮评测范围尚未覆盖 kw
                在约束路径上的池内重排；改动约束路径前，需要先构造一份带版本限定的 Golden
                Query。这里还要区分同一轮内 flag-off 与 flag-on 之间的比特级不变量，以及跨天约 ±3
                条查询的漂移。二者含义不同，但在当前结果中都成立。
              </>
            }
          />
        </p>
      </div>
    </Section>
  )
}
