import { C, M, T } from '../shared'

export default function Abstract() {
  return (
    <section id="abstract" className="mt-8 scroll-mt-24">
      <div className="rounded-md bg-ds-background-200 px-5 py-6 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)] sm:px-8 sm:py-7">
        <div className="mb-2.5 text-sm font-semibold leading-5 tracking-[-0.28px] text-ds-gray-1000">
          <T en="Abstract" zh="摘要" />
        </div>
        <p className="text-sm leading-[1.9] text-ds-gray-900 text-pretty">
          <T
            en={
              <>
                The difficulty of enterprise support-ticket retrieval goes beyond text matching:
                query shapes are heterogeneous. Users type free text, and they also embed ticket
                IDs, version numbers (such as <C>3.x</C>), error strings, or a rough symptom
                description. Under this traffic distribution, general-purpose semantic RAG misses
                exact clues and struggles to express domain rules such as &ldquo;version correctness
                outranks semantic similarity&rdquo;. This paper builds graph-constrained hybrid
                retrieval on a ticket-centric knowledge graph: a rule-based classifier first types
                each query; five channels — issue vectors, solution vectors, full text, the keyword
                graph, and graph relevance (gr) — recall and rank candidates; and
                query-type-weighted Reciprocal Rank Fusion (<M t="k=60" />) merges them. Version
                numbers parsed from the query text hard-slice the <C>Version</C> subgraph; API
                filters in the default soft mode contribute post-fusion bounded boosts only; on the
                result side, external links referenced by the returned tickets are aggregated by
                consensus. gr is the paper&rsquo;s main increment: it uses rarity-weighted
                Link∪Keyword co-citation as a structural signal, participates in re-ranking only,
                and joins through two-pass fusion (base channels fuse first, gr enters the second
                pass); while inactive, output stays bit-identical to the baseline. On 149 Golden
                Queries frozen from real traffic with labels decoupled from the scoring algorithm,
                the gr main effect lifts Hit@1 from 0.362 to 0.463 (+10.1pp, paired McNemar{' '}
                <M t="p=0.006" />) and MRR@10 by 6.1pp. Ablations show gr&rsquo;s gain comes from
                pure re-ranking rather than deeper recall. In same-corpus comparisons against
                general-purpose RAG baselines, this design beats LightRAG hybrid on every quality
                metric (Hit@1 0.772 vs 0.685, MRR@10 0.856 vs 0.764) at roughly 1/11 of its
                indexing-token cost and 1/5 of its median query latency; LightRAG also trails a
                naive vector baseline on the same embedding stack, showing that on a corpus of
                self-contained single documents a generic entity graph&rsquo;s gains rarely cover
                its cost. A production full-corpus rerun (2,521 tickets) further shows naive
                vectors&rsquo; Hit@1 falling to 0.242 as distractors grow — decaying about twice as
                fast as this system (0.490); a nested-subset decay curve (9 sizes × 3 seeds) shows
                that decay is strictly linear in <M t="\ln N" /> (<M t="R^2=0.997" />, −16.7pp per
                corpus doubling, extrapolating to zero near 6,600 tickets), while this
                system&rsquo;s two-point slope is −9.2pp per doubling — the larger the corpus, the
                clearer the advantage of vertical structure.
              </>
            }
            zh={
              <>
                企业支持工单检索的难点不只是文本匹配，而是查询形态混杂。用户可能输入自由文本，也可能夹带工单号、版本号（如{' '}
                <C>3.x</C>
                ）、报错串或只有大致症状的描述。在这种流量分布下，通用语义 RAG
                容易漏掉精确线索，也很难表达&ldquo;版本正确性优先于语义相似&rdquo;这类领域规则。本文在工单中心的知识图谱上构建图约束混合检索：先用规则分类器判断查询类型，再分别从
                issue 向量、solution
                向量、全文、关键词图、图关联（gr）五条通道召回并排名，随后用按查询类型加权的
                Reciprocal Rank Fusion（
                <M t="k=60" />
                ）融合。查询文本中的版本号用于硬切 <C>Version</C> 子图；API
                过滤在默认软模式下只提供后置有界 boost；结果侧再把返回工单引用的外链做共识聚合。gr
                是本文的主要增量，它用稀有度加权的 Link∪Keyword
                共引作为结构信号，只参与重排，经两遍融合（先融合基础通道，再把 gr
                纳入第二次融合）接入；未激活时输出与基线完全一致（比特级相同）。在 149
                条从真实流量冻结、且标签与打分算法解耦的 Golden Query 上，gr 主效应将 Hit@1 从 0.362
                提升到 0.463（+10.1pp，配对 McNemar 检验 <M t="p=0.006" />
                ），MRR@10 提升 6.1pp。消融结果表明，gr 的收益来自纯重排，不是更深召回。与通用 RAG
                基线的同语料对比中，本方案在全部质量指标上超过 LightRAG hybrid（Hit@1 0.772 vs
                0.685，MRR@10 0.856 vs 0.764），索引 token 成本约为其 1/11、查询中位延迟约为其
                1/5；LightRAG 亦全面弱于同 embedding 栈的 naive
                向量基线，说明在&ldquo;单工单自包含&rdquo;的语料上，通用实体图的增益难以覆盖其成本。生产全库（2521
                工单）复跑进一步显示，naive 向量的 Hit@1 随干扰项增长跌至
                0.242，衰减速度约为本方案（0.490）的两倍；嵌套子集衰减曲线（9 个规模 × 3
                seeds）表明该衰减对 <M t="\ln N" /> 严格线性（
                <M t="R^2=0.997" />
                ，每翻倍语料 −16.7pp，外推约 6,600 工单归零），而本方案两点斜率仅
                −9.2pp/倍——语料规模越大，垂直结构的优势越明显。
              </>
            }
          />
        </p>
      </div>
    </section>
  )
}
