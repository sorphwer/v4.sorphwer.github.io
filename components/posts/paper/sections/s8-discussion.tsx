import { C, M, P, Section } from '../shared'

export default function S8Discussion() {
  return (
    <Section id="s8" num="08" en="Discussion & Limitations" zh="讨论与局限">
      <P
        en={
          <>
            The current implementation retains a few deliberate trade-offs. iv and sv come from two
            related embeddings of the same ticket, yet at the fusion layer they are counted as two
            independent channels, which gives the semantic signal more voting weight; under the
            symptom type the semantic-to-literal weight ratio is <M t="2.4:0.8=3:1" />, a
            deliberately chosen prior. kw&rsquo;s behavior on the constrained path has not yet been
            evaluated, because this round&rsquo;s Golden Queries are all version-free queries. With{' '}
            <M t="k=60" /> paired with depth 10, rank information is markedly weakened and the
            fusion layer mainly distinguishes &ldquo;present&rdquo; from &ldquo;absent&rdquo; — the
            kw measurements (Conclusion 3) turned this into a measurable constraint: a single
            channel&rsquo;s rank advantage is not enough to produce a rank-1 flip, and the next jump
            in Hit@1 lives in the fusion layer (score-aware fusion, or using query-side keyword
            anchors as gr evidence features).
          </>
        }
        zh={
          <>
            当前实现保留了几处明确取舍。iv 与 sv
            来自同一张工单的两个相关嵌入，却在融合层被当作两条独立通道分别计票，这会给语义信号更高票权；在
            symptom 类型下，语义与字面的权重比为 <M t="2.4:0.8=3:1" />
            ，这是有意采用的先验设定。kw 在约束路径上的表现尚未评测，因为本轮 Golden Query
            全是无版本约束查询。
            <M t="k=60" /> 配合深度 10 后，名次信息被明显弱化，融合层主要区分“到场”与“缺席”——kw
            通道的实测（结论三）把这一点变成了可测量的约束：单通道名次优势不足以带来 rank-1
            翻转，Hit@1 的下一跳位于融合层（分数感知融合，或把查询侧关键词锚点作为 gr 的证据特征）。
          </>
        }
      />
      <P
        en={
          <>
            A few engineering boundaries also affect later versions. strict version filtering is
            decided on raw strings, the error lexicon is hardcoded English, and non-English error
            strings are classified as the symptom type. gr plays two roles at once — discovery and
            consensus re-ranking: when other seeds are present among the candidates it performs
            consensus re-ranking, otherwise it degrades to tail discovery, and its behavioral
            boundary shifts with <C>GRAPH_SEED_L</C>. Subsequent calibration should tune the{' '}
            <C>confidence</C> thresholds first, then the shape, and only after the first two are
            confirmed stable should it evaluate whether to add signals; every step must pass the
            R1–R3 admission gates.
          </>
        }
        zh={
          <>
            还有一些工程边界会影响后续版本。strict
            版本过滤以原始串判定，报错词表采用纯英文硬编码，非英文报错归入 symptom 类型。gr
            同时承担发现与共识重排两种角色：候选中含其它 seed
            时做共识重排，否则退化为尾部发现，其行为边界会随 <C>GRAPH_SEED_L</C>{' '}
            移动。后续校准应先调 <C>confidence</C>{' '}
            阈值，再调形状，确认前两项稳定后再评估是否增加信号；每一步都需要通过 R1–R3 准入闸门。
          </>
        }
      />
      <P
        en={
          <>
            In the external-baseline comparison (
            <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
              §6
            </a>
            ), the naive vector has completed a full-corpus rerun and a nested-subset decay curve
            (Conclusions 7 and 8: its small-corpus advantage collapses at 2,521 tickets, and its
            decay is linear in <M t="\ln N" />
            ); LightRAG has added a 600-ticket rerun (incremental expansion, 2026-07-15, Conclusion
            6: ranking does not flip, slope between kg and naive). Filling the middle points of this
            system&rsquo;s decay curve requires independently loading a separate Neo4j sub-database
            per scale (channel scoring depends on in-database full statistics, so masked
            re-evaluation is imprecise); together with LightRAG&rsquo;s full-corpus rerun (indexing
            budget about 65M tokens) and the <C>--use-summaries</C> ablation (feeding this
            system&rsquo;s LLM summaries to the baseline, to separate the contribution of
            &ldquo;summary quality&rdquo; from &ldquo;retrieval structure&rdquo;), these remain
            unfinished items.
          </>
        }
        zh={
          <>
            外部基线对比（
            <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
              §6
            </a>
            ）中，naive 向量已完成全库复跑与嵌套子集衰减曲线（结论七、八：其小语料优势在 2521
            工单下崩塌，且衰减对 <M t="\ln N" /> 线性）；LightRAG 已补 600
            工单复跑（增量扩展，2026-07-15，结论六：排序不翻转、斜率介于 kg 与 naive
            之间）。本系统的衰减曲线补齐中间点需要按规模独立灌 Neo4j
            子库（通道打分依赖库内全量统计，掩码复评不精确），与 LightRAG 的全量复跑（索引预算约 65M
            tokens）、<C>--use-summaries</C> 消融（把本系统的 LLM
            摘要喂给基线，以分离“摘要质量”与“检索结构”的贡献）同为未完成项。
          </>
        }
      />
    </Section>
  )
}
