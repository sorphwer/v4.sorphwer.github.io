import { C, Chan, Cite, Lead, M, P, Section } from '../shared'

export default function S2Related() {
  return (
    <Section id="s2" num="02" en="Related Work" zh="相关工作">
      <P
        en={
          <>
            <Lead>Sparse and dense retrieval.</Lead> Sparse retrieval is represented by the BM25
            probabilistic relevance framework <Cite n={1} /> and IDF term specificity <Cite n={2} />
            , while dense retrieval is represented by the bi-encoder DPR <Cite n={3} />. This
            paper&rsquo;s <Chan c="ft" /> full-text channel corresponds to the former, and its{' '}
            <Chan c="iv" />/<Chan c="sv" /> dual-vector channels correspond to the latter. We do not
            treat any one route as the sole answer; instead, we let the different channels each cast
            their ranking opinion at the fusion layer.
          </>
        }
        zh={
          <>
            <Lead>稀疏与稠密检索。</Lead>稀疏检索以 BM25 概率相关性框架 <Cite n={1} /> 和 IDF
            词特异性 <Cite n={2} /> 为代表，稠密检索则以双编码器 DPR <Cite n={3} /> 为代表。本文的{' '}
            <Chan c="ft" /> 全文通道对应前者，
            <Chan c="iv" />/<Chan c="sv" />{' '}
            双向量通道对应后者。这里不把某一条路线当成唯一答案，而是让不同通道在融合层各自给出排序意见。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Rank fusion.</Lead> Reciprocal Rank Fusion <Cite n={4} /> uses rank only and does
            not require the raw scores of different rankers to be comparable, which makes it well
            suited to handling vector, full-text, and graph signals in one layer. This paper makes
            two vertical-domain adjustments to RRF: channel weights vary with query type, and it
            separately analyzes the magnitude consequences of <M t="k=60" />. A larger denominator
            flattens the vote-value gap between &ldquo;rank 1&rdquo; and &ldquo;rank 10&rdquo;, so
            the fusion layer cares more about whether a ticket is nominated by several channels at
            once.
          </>
        }
        zh={
          <>
            <Lead>排序融合。</Lead>Reciprocal Rank Fusion <Cite n={4} />{' '}
            只使用名次，不要求不同排序器的原始分数可比，因此适合把向量、全文和图信号放到同一层处理。本文在
            RRF 上做两处垂直域调整：通道权重随查询类型变化，并单独分析 <M t="k=60" />{' '}
            带来的量级后果。较大的分母会压平“第 1 名”和“第 10
            名”的票值差，融合层更关心一张工单是否被多条通道同时提名。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Graph retrieval and GraphRAG.</Lead> GraphRAG <Cite n={5} />, aimed at global
            summarization, uses community detection and multi-hop reasoning for query-focused
            summarization; LightRAG <Cite n={10} /> lightens it into two-level (low-level entities /
            high-level topics) keyword retrieval over an LLM-extracted entity-relation graph, at an
            indexing cost far below GraphRAG, and is the general-purpose graph-RAG competitor
            closest to this paper. This paper&rsquo;s use of the graph is lighter than both: the
            implementation uses one-hop co-citation as a rank-only endorsement, built from
            ticket-level evidence and lightweight ranking rather than LLM entity extraction,
            multi-hop reasoning, chunk-level evidence synthesis, or a heavy reranker (design scope
            in{' '}
            <a href="#s8" className="text-ds-blue-900 hover:underline">
              §8
            </a>
            ). Here the graph answers &ldquo;which tickets share niche evidence with high-scoring
            tickets&rdquo; rather than directly generating a &ldquo;why relevant&rdquo; explanation.
            The{' '}
            <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
              same-corpus comparison in §6
            </a>{' '}
            shows that on a corpus like tickets — self-contained single documents — this lightweight
            trade-off wins at both the quality and cost ends.
          </>
        }
        zh={
          <>
            <Lead>图检索与 GraphRAG。</Lead>面向全局摘要的 GraphRAG <Cite n={5} />{' '}
            借助社区检测和多跳推理完成 query-focused summarization；LightRAG <Cite n={10} />{' '}
            将其轻量化为 LLM 抽取的实体-关系图上的双层（低层实体 /
            高层主题）关键词检索，索引成本远低于 GraphRAG，是与本文最接近的通用图 RAG
            竞品。本文的图用法比二者都更轻：实现以一跳共引作为 rank-only
            背书，由工单级证据和轻量排序构成，而非 LLM 实体抽取、多跳推理、chunk 级证据合成或重型
            reranker（设计范围见{' '}
            <a href="#s8" className="text-ds-blue-900 hover:underline">
              §8
            </a>
            ）。图在这里回答的是“哪些工单与高分票共享小众证据”，而不是直接生成“为什么相关”的解释。
            <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
              §6 的同语料对比
            </a>
            表明，在工单这类“单文档自包含”的语料上，这一轻量取舍在质量与成本两端同时占优。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Co-citation and bibliometrics.</Lead> Co-citation <Cite n={6} /> in bibliometrics
            characterizes the relatedness of two documents by their being &ldquo;cited together by
            the same set of papers&rdquo;. This paper transfers that intuition to tickets: if two
            tickets co-cite the same niche <C>Link</C>, or share the same long-tail <C>Keyword</C>,
            they are most likely handling the same kind of problem. Co-citation features are
            weighted by rarity, essentially carrying over IDF&rsquo;s <Cite n={2} /> idea of
            down-weighting high-frequency terms.
          </>
        }
        zh={
          <>
            <Lead>共引与文献计量。</Lead>文献计量学中的 co-citation <Cite n={6} />{' '}
            用“被同一批论文共同引用”来刻画两篇文献的关联。本文把这一直觉迁移到工单上：两张工单若共引同一个小众{' '}
            <C>Link</C>，或共享同一个长尾 <C>Keyword</C>
            ，它们大概率在处理同一类问题。共引特征按稀有度加权，本质上沿用了 IDF <Cite n={2} />{' '}
            对高频项降权的思想。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Industrial hybrid retrieval systems.</Lead> Cerebras&rsquo;s enterprise knowledge
            base{' '}
            <a
              href="https://www.cerebras.ai/blog/how-we-built-our-knowledge-base"
              className="text-ds-blue-900 hover:underline"
            >
              <Cite n={11} />
            </a>{' '}
            is the closest industrial parallel to this paper&rsquo;s retrieval core: multiple
            retrieval signals (full-text, embeddings, IDF, age decay) recall in parallel, are fused
            with RRF at <M t="k=60" />, then reranked by a lightweight LLM reranker. Its ingestion
            side reports the same finding — raw text should not be embedded directly: an LLM first
            distills each Slack thread into a normalized question/summary/resolution document before
            embedding, the same discovery as this paper&rsquo;s issue/solution dual-summary
            embeddings (
            <a href="#s3" className="text-ds-blue-900 hover:underline">
              §3
            </a>
            ). The differences are domain width and where the intelligence lives: that system
            targets heterogeneous company-wide corpora (Slack, wiki, code repositories), embeds LLMs
            into every layer — query planning, reranking, and answer synthesis — and stores
            everything in a single flat embeddings table; this paper targets the ticket vertical,
            keeps the online path LLM-free and deterministically reproducible, and relies on two
            domain signals a flat-table architecture cannot express — cross-ticket structural
            co-citation (<Chan c="gr" />,{' '}
            <a href="#s4-5" className="text-ds-blue-900 hover:underline">
              §4.5
            </a>
            ) and query-level hard version constraints (
            <a href="#s4-2" className="text-ds-blue-900 hover:underline">
              §4.2
            </a>
            ). The blog reports no quantitative evaluation; this paper fills that gap with a frozen
            golden set, channel ablations, and external baselines (
            <a href="#s5" className="text-ds-blue-900 hover:underline">
              §5
            </a>
            –
            <a href="#s6" className="text-ds-blue-900 hover:underline">
              §6
            </a>
            ). The two corroborate each other: the multi-channel + RRF (
            <M t="k=60" />) + rerank skeleton converged independently in industry, while vertical
            structural signals and reproducible evaluation are exactly what the general flat-table
            approach leaves open.
          </>
        }
        zh={
          <>
            <Lead>工业界混合检索系统。</Lead>Cerebras 的企业知识库{' '}
            <a
              href="https://www.cerebras.ai/blog/how-we-built-our-knowledge-base"
              className="text-ds-blue-900 hover:underline"
            >
              <Cite n={11} />
            </a>{' '}
            是与本文检索内核最接近的工业界并行案例：多路检索信号（全文、向量、IDF、时间衰减）并行召回，以{' '}
            <M t="k=60" /> 的 RRF 融合，再经轻量 LLM reranker
            重排。其取入侧报告了同一发现——原始文本不宜直接嵌入：LLM 先把 Slack 线程蒸馏为
            question/summary/resolution 规范化文档再嵌入，与本文的 issue/solution 双摘要嵌入（
            <a href="#s3" className="text-ds-blue-900 hover:underline">
              §3
            </a>
            ）同源。差异在问题域宽度与“智能放在哪一层”：该系统面向全公司异构语料（Slack、wiki、代码库），把
            LLM 嵌入查询规划、重排与答案合成的每一层，存储为单张 embeddings
            平表；本文面向工单垂直域，在线路径零
            LLM、确定可复现，并依赖平表架构无法表达的两类领域信号——跨工单结构共引（
            <Chan c="gr" />，
            <a href="#s4-5" className="text-ds-blue-900 hover:underline">
              §4.5
            </a>
            ）与查询级版本硬约束（
            <a href="#s4-2" className="text-ds-blue-900 hover:underline">
              §4.2
            </a>
            ）。该博客未报告定量评测；本文以冻结 golden 集、通道消融与外部基线（
            <a href="#s5" className="text-ds-blue-900 hover:underline">
              §5
            </a>
            –
            <a href="#s6" className="text-ds-blue-900 hover:underline">
              §6
            </a>
            ）补上这一环。两相对照互为佐证：多通道 + RRF（
            <M t="k=60" />
            ）+ 重排的骨架在工业界独立收敛，而垂直域结构信号与可复现评测正是通用平表方案的空缺。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Evaluation.</Lead> This paper uses nDCG <Cite n={7} /> under binary relevance and
            the paired significance test McNemar <Cite n={8} />. Compared with the work above, this
            paper focuses more on the combination of constraints in the ticket vertical: version as
            a hard constraint, co-citation as a rank-only channel, RRF magnitude calibrated around{' '}
            <M t="k=60" />, and an evaluation set drawn from real traffic and frozen outside the
            algorithm. This retrieval stack targets support tickets rather than general open-domain
            question answering.
          </>
        }
        zh={
          <>
            <Lead>评测。</Lead>本文采用二值相关性下的 nDCG <Cite n={7} /> 与配对显著性检验 McNemar{' '}
            <Cite n={8} />
            。与上述工作相比，本文更关注工单垂直域里的约束组合：版本作为硬约束，共引作为 rank-only
            通道，RRF 量级围绕 <M t="k=60" />{' '}
            校准，评测集来自真实流量并在算法外冻结。这套检索栈面向的是支持工单，而不是通用开放域问答。
          </>
        }
      />
    </Section>
  )
}
