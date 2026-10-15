import {
  C,
  Chan,
  DiagramArrow,
  DiagramNode,
  DiagramPanel,
  Fig,
  H3,
  H4,
  Invariant,
  Lead,
  M,
  MD,
  P,
  Section,
  T,
  TableCaption,
  TableShell,
  Td,
  Th,
} from '../shared'
import { FigBound, FigCocit, FigRarity, FigRRF } from '../figures'

const RRF_SCORE_EN =
  '\\text{rrfScore}(t)= \\begin{cases} 1.0, & q \\text{ contains a ticket ID}\\\\[4pt] \\underbrace{s_{\\text{RRF}}(t)}_{\\text{fusion, §4.4}}\\;+\\;\\underbrace{b_{\\text{ver}}(t)}_{\\text{§4.6}}\\;+\\;\\underbrace{b_{\\text{kw}}(t)}_{\\text{§4.6}}, & \\text{otherwise} \\end{cases}'

const RRF_SCORE_ZH =
  '\\text{rrfScore}(t)= \\begin{cases} 1.0, & q \\text{ 含工单号}\\\\[4pt] \\underbrace{s_{\\text{RRF}}(t)}_{\\text{融合，§4.4}}\\;+\\;\\underbrace{b_{\\text{ver}}(t)}_{\\text{§4.6}}\\;+\\;\\underbrace{b_{\\text{kw}}(t)}_{\\text{§4.6}}, & \\text{否则} \\end{cases}'

// `\allowbreak` after each comma lets the set wrap inside the phone column instead of clipping.
const TAU_SET =
  '\\tau\\in\\{\\text{ticket\\_ref},\\allowbreak\\text{error\\_code},\\allowbreak\\text{version\\_scope},\\allowbreak\\text{symptom},\\allowbreak\\text{mixed}\\}'

export default function S4Algorithm() {
  return (
    <Section id="s4" num="04" en="Retrieval Algorithm" zh="检索算法">
      <P
        en={
          <>
            The retrieval pipeline has five layers, and data flows in one direction only. Each
            ticket&rsquo;s final score is defined as:
          </>
        }
        zh={<>检索流水线分为五层，数据只沿一个方向流动。每张工单的最终分定义为：</>}
      />
      <T en={<MD t={RRF_SCORE_EN} />} zh={<MD t={RRF_SCORE_ZH} />} />

      <P
        en={
          <>
            Throughout, we rely on three <Lead>design invariants</Lead>; they are also the
            boundaries the regression tests must hold:
          </>
        }
        zh={
          <>
            全文反复使用三条<Lead>设计不变量</Lead>，它们也是回归测试应守住的边界：
          </>
        }
      />
      <div className="mt-3 flex flex-col gap-2.5">
        <Invariant
          tag="I1"
          en={
            <>
              <Lead>
                Ticket-ID sentinel — a fixed full score that skips fusion and pins to top.
              </Lead>{' '}
              A ticket-ID hit returns <M t="\text{rrfScore}=1.0" /> directly, completed with zero
              embedding and zero API-key consumption.
            </>
          }
          zh={
            <>
              <Lead>工单号哨兵（命中后跳过融合、直接置顶的固定满分）</Lead> 工单号命中直接返回{' '}
              <M t="\text{rrfScore}=1.0" />
              ，以零 embedding 与 API key 消耗完成。
            </>
          }
        />
        <Invariant
          tag="I2"
          en={
            <>
              <Lead>
                Fusion-layer rank-sufficiency — ranks are the only input, raw channel scores stay
                out.
              </Lead>{' '}
              <M t="s_{\text{RRF}}" /> takes the rank <M t="r_c" /> as its only input; the role of{' '}
              <M t="\sigma_c" /> is confined to producing ranks within a channel. Hence any monotone
              transform of a channel&rsquo;s raw scores leaves the fusion result unchanged, and the
              way the gr channel is wired in (§4.5) is a direct corollary of this invariant.
            </>
          }
          zh={
            <>
              <Lead>融合层 rank-sufficiency（只依赖名次、与通道原始分数无关）</Lead>{' '}
              <M t="s_{\text{RRF}}" /> 以排名 <M t="r_c" /> 作为唯一输入；
              <M t="\sigma_c" />{' '}
              的作用限于在通道内产生名次。因此任意通道原始分的单调变换都不影响融合结果，gr
              通道的接入方式（§4.5）正是这条不变量的直接推论。
            </>
          }
        />
        <Invariant
          tag="I3"
          en={
            <>
              <Lead>Boosts are additive, discrete, post-fusion, bounded.</Lead> Boosts are added{' '}
              <strong className="font-semibold">after</strong> fusion, are nonzero only when the API
              explicitly passes filters in soft mode, and satisfy{' '}
              <M t="b_{\text{ver}}+b_{\text{kw}}\le 0.060" />.
            </>
          }
          zh={
            <>
              <Lead>boost 加性、离散、后置、有界</Lead> boost 在融合
              <strong className="font-semibold">之后</strong>叠加，仅当 API 显式传过滤且为 soft
              模式时非零，且 <M t="b_{\text{ver}}+b_{\text{kw}}\le 0.060" />。
            </>
          }
        />
      </div>

      <Fig
        num={3}
        en={
          <>
            Five-layer pipeline overview, marking the ticket-ID sentinel short-circuit (I1) and the
            branch between the constrained and global paths. Adapted from scoring-algorithm doc §0.
          </>
        }
        zh={
          <>
            五层流水线总览，标注工单号哨兵短路（I1）与约束/全局两条路径的分岔。改编自打分算法文档
            §0。
          </>
        }
      >
        <DiagramPanel>
          <div className="flex flex-col">
            <DiagramNode label="IN">
              <T
                en={
                  <>
                    query <M t="q" /> + API parameters
                  </>
                }
                zh={
                  <>
                    query <M t="q" /> + API 参数
                  </>
                }
              />
            </DiagramNode>
            <DiagramArrow />
            <DiagramNode label="01">
              <T
                en="Query understanding: ticket ID / version / keyword hints / type classification"
                zh="查询理解：工单号 / 版本 / keyword hints / 类型分类"
              />
            </DiagramNode>
            <div className="my-1.5 grid grid-cols-[1fr_auto] items-center gap-2">
              <DiagramArrow note={<T en="ticket ID misses" zh="工单号未命中" />} />
              <DiagramNode emphasis>
                <T
                  en={
                    <>
                      <span className="font-geist-mono text-[10.5px] opacity-70">
                        ticket ID hits → I1 sentinel
                      </span>
                      <br />
                      rrfScore = 1.0, returned directly
                    </>
                  }
                  zh={
                    <>
                      <span className="font-geist-mono text-[10.5px] opacity-70">
                        工单号命中 → I1 哨兵
                      </span>
                      <br />
                      rrfScore = 1.0，直接返回
                    </>
                  }
                />
              </DiagramNode>
            </div>
            <DiagramNode label="02">
              <T
                en="Candidate pool: text version → hard-constrained pool · large-pool three gates 300 / 200 / 1000"
                zh="候选池：文本版本 → 硬约束池 · 大池三闸 300 / 200 / 1000"
              />
            </DiagramNode>
            <DiagramArrow note={<T en="version constraint?" zh="有版本约束？" />} />
            <div className="grid grid-cols-2 gap-2.5">
              <DiagramNode>
                <span className="block">
                  <span className="mb-0.5 block font-geist-mono text-[10.5px] text-ds-gray-800">
                    <T en="03A · constrained path (yes)" zh="03A · 约束路径（是）" />
                  </span>
                  <T en="iv · sv · ft · kw (pool re-rank)" zh="iv · sv · ft · kw（池内重排）" />
                </span>
              </DiagramNode>
              <DiagramNode>
                <span className="block">
                  <span className="mb-0.5 block font-geist-mono text-[10.5px] text-ds-gray-800">
                    <T en="03B · global path (no)" zh="03B · 全局路径（否）" />
                  </span>
                  <T
                    en="iv · sv · ft · kw ＋ gr (two-pass fusion)"
                    zh="iv · sv · ft · kw ＋ gr（两遍融合）"
                  />
                </span>
              </DiagramNode>
            </div>
            <DiagramArrow />
            <DiagramNode label="04">
              <T
                en={
                  <>
                    Weighted RRF (<M t="k=60" />) · fusion pool widened to 5×top_k when soft signals
                    are present
                  </>
                }
                zh={
                  <>
                    加权 RRF（
                    <M t="k=60" />
                    ）· soft 信号在场时融合池放宽 5×top_k
                  </>
                }
              />
            </DiagramNode>
            <DiagramArrow />
            <DiagramNode label="05">
              <T
                en="Post-processing: hard filter → version/keyword soft boost → orderBy → truncate to top_k"
                zh="后处理：硬过滤 → 版本/关键词 soft boost → orderBy → 截断 top_k"
              />
            </DiagramNode>
            <DiagramArrow />
            <DiagramNode label="06">
              <T en="Recommended-link consensus → response" zh="推荐链接共识 → response" />
            </DiagramNode>
          </div>
        </DiagramPanel>
      </Fig>

      <H3 id="s4-1" en="4.1 Query Understanding" zh="4.1 查询理解" />
      <P
        en={
          <>
            <C>parse_query</C> extracts four kinds of information from the query string. The ticket
            ID is attempted by three regexes in order, and once one hits, the sentinel I1 takes
            over. A version number is normalized to <M t="\text{norm}(v)=x.y.z" /> on the condition
            that it matches <C>{'\\bv?(\\d+\\.\\d+\\.\\d+)\\b'}</C> and the major version{' '}
            <M t="x\ge3" />; otherwise it is recorded as <C>unknown</C> and falls back to the
            version family <M t="\text{fam}(v)=x.y" />. There are two hard preconditions here: the
            version number must have all three segments, and only Dify 3.x and above are recognized.
            The keyword hints <M t="H" /> come from character-level word-boundary phrase matching of
            the query against the lexicon. The query type <M t="\tau" /> is decided by a rule-based
            classifier.
          </>
        }
        zh={
          <>
            <C>parse_query</C> 从查询串中提取四类信息。工单号由三个正则按序尝试，一旦命中就走哨兵
            I1。版本号规范化为 <M t="\text{norm}(v)=x.y.z" />
            ，条件是匹配 <C>{'\\bv?(\\d+\\.\\d+\\.\\d+)\\b'}</C> 且主版本 <M t="x\ge3" />
            ；否则记为 <C>unknown</C>，并回退到版本族 <M t="\text{fam}(v)=x.y" />
            。这里有两个硬前提：版本号必须完整三段，并且只承认 Dify 3.x 及以上。keyword hints{' '}
            <M t="H" /> 由词典对查询做字符级词边界短语匹配得到。查询类型 <M t="\tau" />{' '}
            则由规则分类器判定。
          </>
        }
      />
      <P
        en={
          <>
            Since 2026-07 the classifier has dropped the LLM fallback and outputs <M t={TAU_SET} />,
            which then indexes the channel weight matrix. Weights are adjusted only within the
            shallow interval <M t="[0.7,1.5]" />; even when classification errs, the perturbation to
            ranking is bounded.
          </>
        }
        zh={
          <>
            分类器在 2026-07 起移除了 LLM 兜底，输出 <M t={TAU_SET} />
            ，再用该类型查询通道权重矩阵。权重只在 <M t="[0.7,1.5]" />{' '}
            的浅区间内调整；即使分类出错，排序受到的扰动也有边界。
          </>
        }
      />

      <div className="mt-5">
        <TableCaption
          num={2}
          en={
            <>
              Channel weight matrix <M t="w_c=W[\tau][c]" /> (<C>QUERY_TYPE_WEIGHTS</C>). The gr
              channel does not enter this matrix; see §4.5.
            </>
          }
          zh={
            <>
              通道权重矩阵 <M t="w_c=W[\tau][c]" />（<C>QUERY_TYPE_WEIGHTS</C>
              ）。gr 通道不进此矩阵，见 §4.5。
            </>
          }
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T
                  en={
                    <>
                      query type <M t="\tau" />
                    </>
                  }
                  zh={
                    <>
                      查询类型 <M t="\tau" />
                    </>
                  }
                />
              </Th>
              <Th align="right">iv</Th>
              <Th align="right">sv</Th>
              <Th align="right">ft</Th>
              <Th align="right">kw</Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Td>
                <T en="symptom (vague symptoms)" zh="symptom（模糊症状）" />
              </Td>
              <Td align="right" mono strong>
                1.2
              </Td>
              <Td align="right" mono strong>
                1.2
              </Td>
              <Td align="right" mono>
                0.8
              </Td>
              <Td align="right" mono>
                1.0
              </Td>
            </tr>
            <tr>
              <Td>
                <T en="error_code (exact error)" zh="error_code（精确报错）" />
              </Td>
              <Td align="right" mono>
                0.7
              </Td>
              <Td align="right" mono>
                0.7
              </Td>
              <Td align="right" mono strong>
                1.5
              </Td>
              <Td align="right" mono>
                1.0
              </Td>
            </tr>
            <tr>
              <Td>
                <T en="version_scope (version scope)" zh="version_scope（版本域）" />
              </Td>
              <Td align="right" mono>
                1.0
              </Td>
              <Td align="right" mono>
                1.0
              </Td>
              <Td align="right" mono>
                1.2
              </Td>
              <Td align="right" mono>
                1.0
              </Td>
            </tr>
            <tr>
              <Td last>
                <T en="mixed" zh="mixed（混合）" />
              </Td>
              <Td last align="right" mono>
                1.0
              </Td>
              <Td last align="right" mono>
                1.0
              </Td>
              <Td last align="right" mono>
                1.0
              </Td>
              <Td last align="right" mono>
                1.0
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>
      <P
        en={
          <>
            The intent of the weight matrix is plain: vague symptoms rely more on semantics, so
            iv/sv are raised and ft lowered; exact errors rely more on the literal text, so ft is
            raised and iv/sv lowered.
          </>
        }
        zh={
          <>
            权重矩阵的意图很朴素：模糊症状更依赖语义，所以上调 iv/sv、下调
            ft；精确报错更依赖字面，所以上调 ft、下调 iv/sv。
          </>
        }
      />

      <H3 id="s4-2" en="4.2 Candidate Pool" zh="4.2 候选池" />
      <P
        en={
          <>
            The version parsed from the query text hard-slices the candidate pool: when exact{' '}
            <M t="x.y.z" /> tickets exist, the exact version is taken, otherwise it falls back to
            the version family <M t="x.y" />; if it is still empty, an empty result is returned
            before any embedding, together with a fallback suggestion. This step is independent of
            the soft boost of the API <C>filter.versions</C> (§4.6): the former decides who may
            enter, the latter only nudges within the arena. Before entering the channels, the
            constrained pool passes three gates in order (three thresholds that throttle by
            candidate size): <M t="n>1000" /> refuses to run, <M t="n>300" /> first does a
            keyword-literal pre-selection to the top 200, otherwise the whole pool is re-ranked
            directly (constants in Appendix A). When no version constraint is given, the query takes
            the global path and the candidate pool is the whole corpus.
          </>
        }
        zh={
          <>
            从查询文本解析出的版本会硬切候选池：有精确 <M t="x.y.z" />{' '}
            工单时取精确版本，否则回退到版本族 <M t="x.y" />
            ；若仍为空，则在 embedding 之前直接返回空结果并给出 fallback 建议。这一步与 API{' '}
            <C>filter.versions</C> 的 soft
            boost（§4.6）相互独立，前者决定谁能进场，后者只在场内轻推。约束池进入通道前依次经过三道闸（按候选规模限流的三个阈值）：
            <M t="n>1000" /> 拒绝执行，
            <M t="n>300" /> 先做关键词字面预选取前 200，否则整池直接重排（常量见附录
            A）。未给定版本约束时，查询走全局路径，候选池为全库。
          </>
        }
      />

      <H3 id="s4-3" en="4.3 The Five Channels" zh="4.3 五通道" />
      <P
        en={
          <>
            The five channels use only the raw score <M t="\sigma_c" /> to sort within a channel
            (see I2); their depths and applicable paths are listed below:
          </>
        }
        zh={
          <>
            五条通道只用原始分 <M t="\sigma_c" /> 在通道内部排序（见
            I2），各自深度与适用路径如下表：
          </>
        }
      />

      <div className="mt-5">
        <TableCaption
          num={3}
          en={<>The five channels at a glance. gr appears only on the global path, at depth 50.</>}
          zh={<>五通道一览。gr 只在全局路径出现，深度 50。</>}
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="channel" zh="通道" />
              </Th>
              <Th>
                <T
                  en={
                    <>
                      raw score <M t="\sigma_c" />
                    </>
                  }
                  zh={
                    <>
                      原始分 <M t="\sigma_c" />
                    </>
                  }
                />
              </Th>
              <Th align="right">
                <T en="depth" zh="深度" />
              </Th>
              <Th align="center">
                <T en="global" zh="全局" />
              </Th>
              <Th align="center">
                <T en="constrained" zh="约束" />
              </Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Td>
                <Chan c="iv" /> issue_vector
              </Td>
              <Td>
                <T en="cosine similarity" zh="余弦相似度" />
              </Td>
              <Td align="right" mono>
                10
              </Td>
              <Td align="center">✓</Td>
              <Td align="center">✓</Td>
            </tr>
            <tr>
              <Td>
                <Chan c="sv" /> solution_vector
              </Td>
              <Td>
                <T en="cosine similarity" zh="余弦相似度" />
              </Td>
              <Td align="right" mono>
                10
              </Td>
              <Td align="center">✓</Td>
              <Td align="center">✓</Td>
            </tr>
            <tr>
              <Td>
                <Chan c="ft" /> fulltext
              </Td>
              <Td>
                <T en="Lucene (BM25 family), opaque" zh="Lucene（BM25 族），不透明" />
              </Td>
              <Td align="right" mono>
                10
              </Td>
              <Td align="center">✓</Td>
              <Td align="center">✓</Td>
            </tr>
            <tr>
              <Td>
                <Chan c="kw" /> keyword_graph
              </Td>
              <Td>
                <T en="IDF sum + activation gate" zh="IDF 和 + 激活闸门" />
              </Td>
              <Td align="right" mono>
                keyword_k
              </Td>
              <Td align="center">✓</Td>
              <Td align="center">✓</Td>
            </tr>
            <tr>
              <Td last>
                <Chan c="gr" /> graph_relevance
              </Td>
              <Td last>
                <T
                  en="Link∪Keyword co-citation + activation gate"
                  zh="Link∪Keyword 共引 + 激活闸门"
                />
              </Td>
              <Td last align="right" mono>
                50
              </Td>
              <Td last align="center">
                ✓
              </Td>
              <Td last align="center" muted>
                ✗
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      <P
        en={
          <>
            The vector channels compute the cosine similarity between the query vector and the
            issue/solution embeddings, e.g.{' '}
            <M t="\sigma_{\text{iv}}(t)=\cos(\vec q,\vec d^{\,\text{issue}}_t)" />; the constrained
            path computes it directly on the Neo4j side with <C>gds.similarity.cosine</C>, guarded
            against empty embeddings and dimension mismatch. ft uses Lucene relevance. The keyword
            channel sums the IDF over the hit hints:
          </>
        }
        zh={
          <>
            向量通道计算查询向量与 issue/solution 嵌入的余弦相似度，例如{' '}
            <M t="\sigma_{\text{iv}}(t)=\cos(\vec q,\vec d^{\,\text{issue}}_t)" />
            ；约束路径用 <C>gds.similarity.cosine</C> 在 Neo4j
            端直算，并带有空嵌入和维度不符防护。ft 使用 Lucene 相关度。关键词通道对命中的 hints 求
            IDF 和：
          </>
        }
      />
      <MD t="\sigma_{\text{kw}}(t)=\sum_{w\in H\cap K_t}\ln\frac{N}{df_w}" />
      <P
        en={
          <>
            This channel has an activation gate: only when{' '}
            <M t="\max_t\sigma_{\text{kw}}(t)>\ln 10\approx2.303" /> (strictly greater) does it
            enter fusion, otherwise the whole channel is discarded. If the hits are all
            high-frequency, low-IDF words, discrimination is insufficient; the system admits the kw
            vote only when discrimination is sufficient, avoiding bringing candidates of
            insufficient discrimination into the fusion layer. On the global path, kw is promoted
            from pool re-ranking to a recall channel, its candidates being all tickets in the corpus
            that hit at least one hint keyword, with scoring rules and gate identical to the
            constrained path. Here is a hard invariant: when kw is not requested or hits only
            high-frequency words, the output is bit-identical to the iv/sv/ft baseline.
          </>
        }
        zh={
          <>
            该通道设有激活闸门：只有当 <M t="\max_t\sigma_{\text{kw}}(t)>\ln 10\approx2.303" />
            （严格大于）时才进入融合，否则整条通道丢弃。若命中的全是高频低 IDF
            词，区分度不足，系统仅在区分度充分时纳入 kw
            投票，避免将区分度不足的候选带入融合层。在全局路径上，kw
            从池内重排升格为召回通道，候选是全库中至少命中一个 hint
            关键词的所有工单，打分规则与闸门都与约束路径一致。这里有一条硬不变量：kw
            未请求或仅命中高频词时，输出与 iv/sv/ft 基线完全一致（比特级相同）。
          </>
        }
      />

      <H3 id="s4-4" en="4.4 Weighted Reciprocal Rank Fusion" zh="4.4 加权 Reciprocal Rank Fusion" />
      <P
        en={
          <>
            Each channel produces a rank <M t="r_c(t)" /> within its depth (with{' '}
            <M t="r_c=\infty" /> when the ticket is not in that channel); the fusion score is
          </>
        }
        zh={
          <>
            每条通道在深度内产生名次 <M t="r_c(t)" />
            （不在该通道则 <M t="r_c=\infty" />
            ），融合分为
          </>
        }
      />
      <MD t="s_{\text{RRF}}(t)=\sum_{c\in C(t)}\frac{w_c}{k+r_c(t)},\qquad k=60" />
      <P
        en={
          <>
            The final ordering is a three-key total order{' '}
            <M t="(s_{\text{RRF}},\,-\min_c r_c,\,t_{\text{id}})" /> sorted in descending order.{' '}
            <M t="k=60" /> makes a single channel&rsquo;s contribution within depth 10 nearly a flat
            line (Figure 4), so the fusion layer behaves more like a weighted voting machine:
          </>
        }
        zh={
          <>
            最终按三键全序 <M t="(s_{\text{RRF}},\,-\min_c r_c,\,t_{\text{id}})" /> 降序排序。
            <M t="k=60" /> 会让深度 10 以内的单通道贡献接近一条平线（图
            4），所以融合层更像一台加权投票机：
          </>
        }
      />

      <Fig
        num={4}
        en={
          <>
            Decay of a single channel&rsquo;s contribution <M t="1/(60+r)" /> with rank <M t="r" />:
            from rank 1 (0.0164) to rank 10 (0.0143) it drops only 13%, almost flat, showing that
            the rank difference matters far less than &ldquo;being nominated by several channels at
            once&rdquo;.
          </>
        }
        zh={
          <>
            单通道贡献 <M t="1/(60+r)" /> 随名次 <M t="r" /> 的衰减：从 rank 1（0.0164）到 rank
            10（0.0143）只降了 13%，几近平坦，可见名次差远不如&ldquo;被几条通道同时提名&rdquo;重要。
          </>
        }
      >
        <FigRRF />
      </Fig>

      <div className="mt-6">
        <TableCaption
          num={4}
          en={
            <>
              One ruler: the comparable magnitudes of the various score differences (<M t="k=60" />
              ).
            </>
          }
          zh={
            <>
              同一把尺子：各类分差的可比量级（
              <M t="k=60" />
              ）。
            </>
          }
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="event" zh="事件" />
              </Th>
              <Th align="right">
                <T
                  en={
                    <>
                      contribution to <M t="s_{\text{RRF}}" />
                    </>
                  }
                  zh={
                    <>
                      对 <M t="s_{\text{RRF}}" /> 的贡献
                    </>
                  }
                />
              </Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Td>
                <T
                  en={<>a channel going from &ldquo;absent&rdquo; to rank 1 in that channel</>}
                  zh={<>某通道从&ldquo;缺席&rdquo;到该通道 rank 1</>}
                />
              </Td>
              <Td align="right" mono>
                <M t="1/61 = 0.0164" />
              </Td>
            </tr>
            <tr>
              <Td>
                <T
                  en="rising from rank 10 to rank 1 within a channel"
                  zh="某通道内从 rank 10 升到 rank 1"
                />
              </Td>
              <Td align="right" mono>
                <M t="1/61-1/70 = 0.0021" />
              </Td>
            </tr>
            <tr>
              <Td>
                <T en="exact version-hit boost (§4.6)" zh="版本精确命中 boost（§4.6）" />
              </Td>
              <Td align="right" mono>
                <M t="0.030" />
              </Td>
            </tr>
            <tr>
              <Td last>
                <T
                  en="upper bound of a gr-only newcomer's fusion contribution (§4.5)"
                  zh="gr-only 新票融合贡献上界（§4.5）"
                />
              </Td>
              <Td last align="right" mono>
                <M t="0.7/61 = 0.0115" />
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>
      <P
        en={
          <>
            These magnitudes show that a ticket gaining one more channel nomination (+0.0164) has
            more impact than moving it from rank 10 to rank 1 within a single channel (+0.0021); the
            version boost (+0.030) can flip an adjacent vote but usually cannot beat the consistent
            nomination of two channels. This is exactly what <M t="k=60" /> does: it tunes the
            fusion layer to &ldquo;prefer multi-source agreement&rdquo; rather than over-chasing
            single-channel rank.
          </>
        }
        zh={
          <>
            这些量级说明，一张工单多得到一条通道提名（+0.0164），比它在单条通道内从 rank 10 挪到
            rank 1（+0.0021）更有影响；版本
            boost（+0.030）能翻转相邻票，但通常压不过两条通道的一致提名。
            <M t="k=60" />{' '}
            的作用就在这里：把融合层调成&ldquo;多源一致优先&rdquo;，而不是过度追逐单通道名次。
          </>
        }
      />

      <H3 id="s4-5" en="4.5 The Graph-Relevance Channel gr" zh="4.5 图关联通道 gr" />
      <P
        en={
          <>
            The intuition of gr is simple: if two tickets cite the same niche document or share the
            same long-tail keyword, they are most likely handling the same matter. gr is the fifth,
            rank-only channel: it takes the high-scoring votes after the first fusion pass as seeds,
            then looks for tickets that co-cite the same niche feature (a Link or a Keyword) with
            these seeds, giving them an independent rank support.
          </>
        }
        zh={
          <>
            gr
            的直觉很简单：两张工单若引用同一个小众文档，或共享同一个长尾关键词，多半在处理同一件事。gr
            是第五条 rank-only 通道，它把第一遍融合后的高分票作为 seed，再寻找与这些 seed
            共引同一小众特征（Link 或 Keyword）的工单，为它们提供一条独立名次支持。
          </>
        }
      />

      <H4
        en="4.5.1 Motivation: two gaps and co-citation density"
        zh="4.5.1 动机：两个缺口与共引密度"
      />
      <P
        en={
          <>
            gr supplements retrieval with structural evidence. Link co-citation yields 881 ticket
            pairs, too few to form a stable channel; after merging in the Keyword edge type, ticket
            pairs sharing at least one Keyword reach 747,636 (about 850×), a co-citation density
            sufficient to support retrieval:
          </>
        }
        zh={
          <>
            gr 为检索补充结构证据。Link 共引的工单对为 881 对，难以形成稳定通道；并入 Keyword
            边型后，共享至少一个 Keyword 的工单对达到 747,636（约 850 倍），共引密度足以支撑检索：
          </>
        }
      />
      <Fig
        num={5}
        en={
          <>
            Co-citation density (<M t="N=2486" />, 2026-07-06, logarithmic y-axis): ticket pairs
            sharing ≥1 Link number only 881, while pairs sharing ≥1 Keyword reach 747,636 —
            precisely the direct motivation for merging the Keyword edge type into gr.
          </>
        }
        zh={
          <>
            共引密度（
            <M t="N=2486" />
            ，2026-07-06，对数纵轴）：共享 ≥1 个 Link 的工单对仅 881，共享 ≥1 个 Keyword 的对达
            747,636，这正是把 Keyword 边型并入 gr 的直接动机。
          </>
        }
      >
        <FigCocit />
      </Fig>

      <H4 en="4.5.2 Scoring: rarity-weighted co-citation" zh="4.5.2 打分：稀有度加权共引" />
      <P
        en={
          <>
            Seed strength is normalized by the first-pass fusion score,{' '}
            <M t="p(s)=s^{(1)}_{\text{RRF}}(s)/\max_{s'}s^{(1)}_{\text{RRF}}(s')\in(0,1]" />
            . The gr raw score of a candidate <M t="t" /> is accumulated over all shared features:
          </>
        }
        zh={
          <>
            seed 强度按第一遍融合分归一化，{' '}
            <M t="p(s)=s^{(1)}_{\text{RRF}}(s)/\max_{s'}s^{(1)}_{\text{RRF}}(s')\in(0,1]" />
            。候选 <M t="t" /> 的 gr 原始分由所有共享特征累加得到：
          </>
        }
      />
      <MD t="\sigma_{gr}(t)=\sum_{s\ne t}\ \sum_{u\in\text{feats}(s)\cap\text{feats}(t)} p(s)\cdot r(u)" />
      <P
        en={
          <>
            Feature rarity is down-weighted by ticket-degree, equivalent to transplanting the idea
            of IDF onto graph features:
          </>
        }
        zh={<>特征稀有度按 ticket-degree 降权，相当于把 IDF 的思想迁移到图特征上：</>}
      />
      <MD t="r(u)=\operatorname{clip}\!\left(\frac{\ln\frac{N+1}{\deg(u)+1}}{\ln(N+1)},\,0,\,1\right)" />
      <P
        en={
          <>
            Keyword rarity has a floor <M t="r(k)\ge0.5" />, because keywords are naturally more
            frequent than Links and without a floor would be crushed altogether; Links have no
            floor. Rarity decreases monotonically with degree, as in Figure 6:
          </>
        }
        zh={
          <>
            Keyword 稀有度设地板 <M t="r(k)\ge0.5" />
            ，因为关键词天然比 Link 高频，不设地板会被整体压没；Link 不设地板。稀有度随 degree
            单调下降，如图 6：
          </>
        }
      />
      <Fig
        num={6}
        en={
          <>
            Feature rarity <M t="r(u)" /> decreases with ticket-degree <M t="\deg(u)" />: rare
            features (deg 2–3) approach full weight, while high-frequency features (deg 400) drop to
            about 0.20.
          </>
        }
        zh={
          <>
            特征稀有度 <M t="r(u)" /> 随 ticket-degree <M t="\deg(u)" /> 递减：冷门特征（deg
            2–3）接近满权，高频特征（deg 400）降到约 0.20。
          </>
        }
      >
        <FigRarity />
      </Fig>

      <P
        en={
          <>
            gr has three activation conditions: <M t="|S|\ge3" />, the existence of scored
            candidates, and <M t="\max_t\sigma_{gr}>0.25" />. It uses a single weight{' '}
            <M t="w_{gr}=0.7" />, does not enter <M t="W[\tau]" />, and is wired in through two-pass
            fusion. While inactive or with the flag off, the output is bit-identical to the baseline
            without gr.
          </>
        }
        zh={
          <>
            gr 有三条激活条件：
            <M t="|S|\ge3" />
            、存在有得分的候选、以及 <M t="\max_t\sigma_{gr}>0.25" />
            。它使用单一权重 <M t="w_{gr}=0.7" />
            ，不进入 <M t="W[\tau]" />
            ，并通过两遍融合接入。在未激活或 flag 关闭的状态下，输出与不含 gr
            的基线完全一致（比特级相同）。
          </>
        }
      />

      <H4
        en="4.5.3 Safety ceiling: gr stays structurally outside the serving window"
        zh="4.5.3 安全上界：gr 结构性进不了服务窗口"
      />
      <div className="mt-3">
        <Invariant
          tag="P7"
          en={
            <>
              <Lead>gr-only safety ceiling.</Lead> A newcomer{' '}
              <strong className="font-semibold">brought in by gr alone</strong> has, in the second
              fusion pass, a contribution bounded by <M t="w_{gr}/(k+1)=0.7/61\approx0.0115" />,
              below the typical fusion score of an incumbent: a single channel at rank 1 already
              gives 0.0164, two channels 0.0328, three channels 0.0492. Therefore the maximum
              contribution of a gr-only newcomer stays below the serving-window threshold. On
              candidates that <strong className="font-semibold">contain other seeds</strong>, gr
              acts as consensus re-ranking; when a ticket is brought in by gr alone, the serving
              result stays unchanged.
            </>
          }
          zh={
            <>
              <Lead>gr-only 安全上界</Lead> 一张
              <strong className="font-semibold">仅由 gr 带入</strong>
              的新票，在二次融合中的贡献上界为 <M t="w_{gr}/(k+1)=0.7/61\approx0.0115" />
              ，低于在位票的典型融合分：单通道 rank 1 已有 0.0164，双通道为 0.0328，三通道为
              0.0492。因此，gr-only 新票的最大贡献保持在服务窗口阈值以下。gr 在
              <strong className="font-semibold">含其它 seed</strong>
              的候选上表现为共识重排；仅由 gr 带入时，服务结果保持不变。
            </>
          }
        />
      </div>
      <Fig
        num={7}
        en={
          <>
            Safety-ceiling comparison: the gr-only newcomer&rsquo;s ceiling of 0.0115 is below a
            single-channel incumbent&rsquo;s rank 1 (0.0164), and far below the consistent
            nomination of two or three channels; hence gr&rsquo;s effective role is the re-ranking
            of already-recalled votes, while gr-only newcomers stay outside the serving window.
          </>
        }
        zh={
          <>
            安全上界对照：gr-only 新票的上界 0.0115 低于单通道在位票的 rank
            1（0.0164），更远低于双、三通道的一致提名，因此 gr
            的有效作用表现为已召回票的重排，gr-only 新票保持在服务窗口之外。
          </>
        }
      >
        <FigBound />
      </Fig>

      <H3 id="s4-6" en="4.6 Post-processing and Downstream" zh="4.6 后处理与下游" />
      <P
        en={
          <>
            After fusion, the system in order applies hard filtering (priority, status, date,
            handledBy; strict mode adds versions, keywords), the version boost (exact hit{' '}
            <M t="+0.030" />, version family <M t="+0.018" />, bucketed by discrete prefix rather
            than numeric distance), the keyword boost{' '}
            <M t="b_{\text{kw}}=\min(0.010\cdot|F_k\cap K_t|,\,0.030)" />, orderBy overrides, and
            truncates to top-
            <M t="k" />. Before returning, the system also aggregates by consensus the external
            links referenced by the result set:
          </>
        }
        zh={
          <>
            融合之后，系统依次执行硬过滤（priority、status、date、handledBy，strict 模式再加
            versions、keywords）、版本 boost（精确命中 <M t="+0.030" />
            、版本族 <M t="+0.018" />
            ，按离散前缀分桶而非数值距离）、关键词 boost{' '}
            <M t="b_{\text{kw}}=\min(0.010\cdot|F_k\cap K_t|,\,0.030)" />
            、orderBy 覆盖，并截断到 top-
            <M t="k" />
            。返回前，系统还会对结果集所引外链做共识聚合：
          </>
        }
      />
      <MD t="\text{consensus}(u)=\Big(\!\!\sum_{t\in R,\,u\in\text{refs}(t)}\!\!\text{rrfScore}(t)\Big)\cdot\log_2\big(1+\text{support}(u)\big)" />
      <P
        en={
          <>
            confidence is synthesized separately as a heuristic signal of absolute trustworthiness,
            kept apart from the relative ranks given by RRF:
          </>
        }
        zh={<>confidence 作为绝对可信度的启发式信号单独合成，与 RRF 给出的相对名次分开：</>}
      />
      <MD t="\text{confidence}(t)=\frac{\text{channelCoverage}(t)+\text{topMargin}(t)}{2},\quad \text{queryConfidence}=\max_{t\in R}\text{confidence}(t)" />
      <P
        en={
          <>
            The grading thresholds are 0.65 and 0.40.{' '}
            <M t="\text{missedRetrieval}=\max\!\big(0,\ \max_{\text{pool}}\text{conf}-\text{queryConfidence}\big)" />{' '}
            is used to observe whether a more trustworthy ticket remains outside the serving window.
            confidence is used for rough cross-query comparison, and stays separated in
            responsibility from RRF&rsquo;s relative ordering, answer-correctness judgment, and
            forced re-ranking.
          </>
        }
        zh={
          <>
            分级阈值取 0.65 与 0.40。{' '}
            <M t="\text{missedRetrieval}=\max\!\big(0,\ \max_{\text{pool}}\text{conf}-\text{queryConfidence}\big)" />{' '}
            用来观察是否有更可信的工单留在服务窗口之外。confidence 用于跨 query 做粗略比较，并与 RRF
            相对排序、答案正确性判定及强制重排保持职责分离。
          </>
        }
      />
    </Section>
  )
}
