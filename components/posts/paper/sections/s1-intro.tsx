import { C, Cite, Contribution, Fig, Lead, M, P, Section, useT } from '../shared'

export default function S1Intro() {
  const t = useT()
  return (
    <Section id="s1" num="01" en="Introduction" zh="引言与背景">
      <Fig
        num={1}
        en={
          <>
            Global system architecture. The top half is the online retrieval pipeline — query
            understanding, five-channel hybrid retrieval (iv/sv/ft/kw/gr with weighted RRF),
            post-processing, and the exit — feeding the Agent, Dify External Knowledge, the CLI, and
            Ops/Web. The bottom half is offline ingestion and infrastructure: Zendesk tickets flow
            through the Batch and Incremental pipelines into the Neo4j knowledge graph that
            retrieval reads from, and the Vercel proxy asynchronously writes call logs to Cloudflare
            D1 via <C>after()</C>. The deployment stack is Vercel · GCP Cloud Run · Neo4j AuraDB ·
            Cloudflare D1. Adapted from architecture doc §1.
          </>
        }
        zh={
          <>
            全局系统架构。上半部分为在线检索流水线，依次是查询理解、五通道混合检索（含
            iv/sv/ft/kw/gr 与加权 RRF）、后处理与出口，结果供给 Agent、Dify External Knowledge、CLI
            与 Ops/Web。下半部分为离线取入与基础设施：Zendesk 工单经 Batch / Incremental 两条
            pipeline 汇入 Neo4j 知识图谱供检索读取，Vercel 代理经 <C>after()</C> 异步把调用日志写入
            Cloudflare D1。部署栈为 Vercel · GCP Cloud Run · Neo4j AuraDB · Cloudflare
            D1。改编自架构文档 §1。
          </>
        }
      >
        <img
          src="/img/in-post/2026-07-09-hybrid-retrieval-support-ticket/paper_arch.png"
          alt={t(
            'Global system architecture: the top half is the online retrieval pipeline (query understanding \u2192 five-channel hybrid retrieval \u2192 post-processing and exit); the bottom half is offline ingestion and infrastructure (Zendesk flows through two pipelines into the Neo4j knowledge graph), with the Vercel proxy asynchronously writing call logs to Cloudflare D1 via after()',
            '\u5168\u5c40\u7cfb\u7edf\u67b6\u6784\uff1a\u4e0a\u534a\u4e3a\u5728\u7ebf\u68c0\u7d22\u6d41\u6c34\u7ebf\uff08\u67e5\u8be2\u7406\u89e3 \u2192 \u4e94\u901a\u9053\u6df7\u5408\u68c0\u7d22 \u2192 \u540e\u5904\u7406\u4e0e\u51fa\u53e3\uff09\uff0c\u4e0b\u534a\u4e3a\u79bb\u7ebf\u53d6\u5165\u4e0e\u57fa\u7840\u8bbe\u65bd\uff08Zendesk \u7ecf\u4e24\u6761 pipeline \u6c47\u5165 Neo4j \u77e5\u8bc6\u56fe\u8c31\uff09\uff0cVercel \u4ee3\u7406\u7ecf after() \u5f02\u6b65\u5199\u8c03\u7528\u65e5\u5fd7\u5230 Cloudflare D1'
          )}
          className="block max-w-full rounded-md shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400),0_2px_2px_rgba(0,0,0,.04)]"
        />
      </Fig>

      <P
        en={
          <>
            <Lead>Problem domain.</Lead> This system targets Dify enterprise support-ticket
            retrieval; its consumers include human support Agents and the{' '}
            <C>Dify External Knowledge</C> external-knowledge interface. Queries are free-form user
            input and may contain a ticket ID (<C>ticket #12345</C>), a version number (<C>3.4.1</C>
            ), an error string (<C>ECONNREFUSED</C>), or just a one-line symptom description. Such
            traffic is unfriendly to any single semantic channel: a short error string is close to
            noise inside a semantic encoder, version numbers have low discriminative power in vector
            space, and the answer to one bug is often scattered across several historical tickets.
            The system goals and the knowledge-graph scope follow the existing conventions of{' '}
            <Cite n={9} />.
          </>
        }
        zh={
          <>
            <Lead>问题域。</Lead>本系统面向 Dify 企业支持工单检索，消费方包括人工客服 Agent 和{' '}
            <C>Dify External Knowledge</C> 外部知识接口。查询由用户自由输入，可能包含工单号（
            <C>ticket #12345</C>）、版本号（<C>3.4.1</C>）、报错串（
            <C>ECONNREFUSED</C>
            ），也可能只是一句症状描述。这样的流量对单一语义通道并不友好：短报错串在语义编码器里接近噪声，版本号在向量空间中区分度很低，同一个
            bug 的答案又常散落在多张历史工单里。系统目标与知识图谱范围沿用 <Cite n={9} />{' '}
            的既有约定。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Why this is hard, and why a graph.</Lead> The lexical and semantic channels fill
            two different kinds of gaps. Lexical search catches error strings but struggles to
            connect &ldquo;cannot reach the cache&rdquo; with &ldquo;redis refused the
            connection&rdquo;; semantic search handles near-synonymous phrasings but readily
            overlooks precise clues such as version numbers and error codes. Ticket retrieval also
            carries strong domain constraints: the patch differences between Dify <C>3.4.1</C> and{' '}
            <C>3.5.0</C> can make the semantically closest ticket give a wrong answer. Structural
            evidence cannot be discarded either. Multiple tickets for the same bug frequently cite
            the same obscure troubleshooting document, or share the same set of long-tail keywords,
            and this co-citation relationship is not obvious in either vector or full-text space.
            Zendesk&rsquo;s native lexical search scores almost entirely zero on this paper&rsquo;s
            Golden Queries (Hit@1=0.000,{' '}
            <a href="#tbl-main" className="text-ds-blue-900 hover:underline">
              see the §6 main table
            </a>
            ); although it is not a fair control baseline, it is enough to show the mismatch between
            generic lexical mechanisms and real natural-language queries.
          </>
        }
        zh={
          <>
            <Lead>难点与用图的理由。</Lead>
            字面通道和语义通道补足的是两类不同缺口。字面搜索能抓住报错串，却很难把“连不上缓存”和“redis
            拒绝连接”连到一起；语义搜索能处理近义表达，却容易忽略版本号、错误码这类精确线索。工单检索还带有强领域约束：Dify{' '}
            <C>3.4.1</C> 与 <C>3.5.0</C>{' '}
            的补丁差异，可能让语义最接近的工单给出错误答案。结构证据也不能丢掉。同一个 bug
            的多张工单经常引用同一份冷门排障文档，或共享同一组长尾关键词，这种共引关系在向量与全文空间里都不明显。Zendesk
            原生字面搜索在本文 Golden Query 上几乎全零命中（Hit@1=0.000，
            <a href="#tbl-main" className="text-ds-blue-900 hover:underline">
              见 §6 主表
            </a>
            ），虽然它不是公平的对照基线，但足以说明通用字面机制与真实自然语言查询之间存在错配。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>This paper is organized around five contributions</Lead>:
          </>
        }
        zh={
          <>
            <Lead>本文围绕以下五点展开</Lead>：
          </>
        }
      />

      <div className="mt-3 flex flex-col gap-2.5">
        <Contribution
          tag="C1"
          en={
            <>
              A ticket-centric knowledge graph and a five-layer graph-constrained hybrid-retrieval
              architecture that treats version as a hard constraint and structural evidence as an
              independent signal (
              <a href="#s3" className="text-ds-blue-900 hover:underline">
                §3
              </a>
              ,{' '}
              <a href="#s4" className="text-ds-blue-900 hover:underline">
                §4
              </a>
              ).
            </>
          }
          zh={
            <>
              一套工单中心的知识图谱与五层图约束混合检索架构，将版本作为硬约束、把结构证据作为独立信号（
              <a href="#s3" className="text-ds-blue-900 hover:underline">
                §3
              </a>
              、
              <a href="#s4" className="text-ds-blue-900 hover:underline">
                §4
              </a>
              ）。
            </>
          }
        />
        <Contribution
          tag="C2"
          en={
            <>
              Query-type-weighted RRF fusion, whose magnitude analysis of <M t="k=60" /> shows the
              large denominator turns the fusion layer into an approximate weighted voting machine —
              a ticket nominated by several channels at once carries far more weight than one ranked
              first by a single channel (
              <a href="#s4-4" className="text-ds-blue-900 hover:underline">
                §4.4
              </a>
              ) — and a rank-only (depending only on rank, independent of channels&rsquo; raw
              scores) graph-relevance channel gr, signaled by rarity-weighted Link∪Keyword
              co-citation, with a safety upper-bound proof and the engineering invariant of
              &ldquo;two-pass fusion, output bit-identical to the baseline while inactive&rdquo; (
              <a href="#s4-5" className="text-ds-blue-900 hover:underline">
                §4.5
              </a>
              ).
            </>
          }
          zh={
            <>
              按查询类型加权的 RRF 融合：对 <M t="k=60" />{' '}
              的量级分析表明，大分母使融合层近似一台加权投票机，被多条通道同时提名的工单，其权重远高于只被单条通道排到第一的工单（
              <a href="#s4-4" className="text-ds-blue-900 hover:underline">
                §4.4
              </a>
              ）；以及以稀有度加权的 Link∪Keyword 共引为信号的
              rank-only（只依赖名次、与通道原始分数无关）图关联通道
              gr，附安全上界证明，以及“两遍融合、未激活时输出与基线完全一致（比特级相同）”的工程不变量（
              <a href="#s4-5" className="text-ds-blue-900 hover:underline">
                §4.5
              </a>
              ）。
            </>
          }
        />
        <Contribution
          tag="C3"
          en={
            <>
              A golden evaluation methodology mined and frozen from real traffic, with labels
              decoupled from the scoring algorithm (
              <a href="#s5" className="text-ds-blue-900 hover:underline">
                §5
              </a>
              ), and on top of it a 2×2 kw×gr ablation that quantifies gr&rsquo;s main effect,
              proves its gain comes from pure re-ranking, and locates kw&rsquo;s contribution in the
              serving window and confidence (
              <a href="#s6" className="text-ds-blue-900 hover:underline">
                §6
              </a>
              ).
            </>
          }
          zh={
            <>
              一套从真实流量挖掘并冻结、且标签与打分算法解耦的 golden 评测方法（
              <a href="#s5" className="text-ds-blue-900 hover:underline">
                §5
              </a>
              ），并在其上以 kw×gr 的 2×2 消融实验，量化 gr 的主效应，证明其增益来自纯重排，并定位
              kw 的贡献在服务窗口与置信度（
              <a href="#s6" className="text-ds-blue-900 hover:underline">
                §6
              </a>
              ）。
            </>
          }
        />
        <Contribution
          tag="C4"
          en={
            <>
              A same-corpus retrieval-layer comparison with general-purpose RAG baselines (LightRAG
              hybrid, naive vector retrieval) and an indexing/query cost accounting: the vertical
              design wins on every quality metric at roughly 1/11 of LightRAG&rsquo;s indexing cost
              (
              <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
                §6 external-baselines subsection
              </a>
              ).
            </>
          }
          zh={
            <>
              与通用 RAG 基线（LightRAG hybrid、naive
              向量检索）的同语料检索层对比与索引/查询成本核算：垂直方案在全部质量指标上胜出，索引成本约为
              LightRAG 的 1/11（
              <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
                §6 外部基线小节
              </a>
              ）。
            </>
          }
        />
        <Contribution
          tag="C5"
          en={
            <>
              An engineering implementation based on a single pushed-down Cypher query, a
              second-scale latency budget, and zero-cost A/B and rollback (
              <a href="#s7" className="text-ds-blue-900 hover:underline">
                §7
              </a>
              ).
            </>
          }
          zh={
            <>
              基于单条 Cypher 下推的工程实现，秒级延迟预算，以及零成本的 A/B 与回滚（
              <a href="#s7" className="text-ds-blue-900 hover:underline">
                §7
              </a>
              ）。
            </>
          }
        />
      </div>

      <P
        en={
          <>
            <Lead>Key findings at a glance</Lead> (full experiments in §6):
          </>
        }
        zh={
          <>
            <Lead>主要发现</Lead>（完整实验见 §6）：
          </>
        }
      />

      <div className="mt-3 flex flex-col gap-2.5">
        <Contribution
          tag="F1"
          en={
            <>
              Zendesk&rsquo;s native search scores near zero on the golden set — Hit@1{' '}
              <span className="font-semibold text-ds-gray-1000">0.000</span>, Hit@5{' '}
              <span className="font-semibold text-ds-gray-1000">0.027</span>, p50 26.4s — while the
              full pipeline reaches Hit@1{' '}
              <span className="font-semibold text-ds-gray-1000">0.450</span> / Hit@5{' '}
              <span className="font-semibold text-ds-gray-1000">0.906</span> at sub-second p50,
              roughly <span className="font-semibold text-ds-gray-1000">30×</span> faster (
              <a href="#tbl-main" className="text-ds-blue-900 hover:underline">
                §6 main table
              </a>
              ).
            </>
          }
          zh={
            <>
              Zendesk 原生搜索在 golden 集上近乎为零——Hit@1{' '}
              <span className="font-semibold text-ds-gray-1000">0.000</span>、Hit@5{' '}
              <span className="font-semibold text-ds-gray-1000">0.027</span>、p50
              26.4s——而完整流水线达到 Hit@1{' '}
              <span className="font-semibold text-ds-gray-1000">0.450</span> / Hit@5{' '}
              <span className="font-semibold text-ds-gray-1000">0.906</span>，p50 亚秒级，约快{' '}
              <span className="font-semibold text-ds-gray-1000">30×</span>（
              <a href="#tbl-main" className="text-ds-blue-900 hover:underline">
                §6 主表
              </a>
              ）。
            </>
          }
        />
        <Contribution
          tag="F2"
          en={
            <>
              The gr channel lifts Hit@1 0.362→
              <span className="font-semibold text-ds-gray-1000">0.463</span> (
              <span className="font-semibold text-ds-gray-1000">+10.1pp</span>, p=.006) by pure
              re-ranking — the gold-ticket recall set stays unchanged (
              <a href="#s6" className="text-ds-blue-900 hover:underline">
                §6
              </a>
              ).
            </>
          }
          zh={
            <>
              gr 通道以纯重排将 Hit@1 从 0.362→
              <span className="font-semibold text-ds-gray-1000">0.463</span>（
              <span className="font-semibold text-ds-gray-1000">+10.1pp</span>
              ，p=.006）——gold 工单的召回集保持不变（
              <a href="#s6" className="text-ds-blue-900 hover:underline">
                §6
              </a>
              ）。
            </>
          }
        />
        <Contribution
          tag="F3"
          en={
            <>
              On the same 300-ticket corpus this system beats LightRAG hybrid on every quality
              metric (Hit@1 <span className="font-semibold text-ds-gray-1000">+8.7pp</span>) at{' '}
              <span className="font-semibold text-ds-gray-1000">1/11.6</span> the indexing LLM
              tokens and <span className="font-semibold text-ds-gray-1000">1/5</span> the query
              latency; LightRAG even loses to naive vector RAG on this corpus shape (
              <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
                §6 external baselines
              </a>
              ).
            </>
          }
          zh={
            <>
              在同一 300 工单语料上，本系统在全部质量指标上胜过 LightRAG hybrid（Hit@1{' '}
              <span className="font-semibold text-ds-gray-1000">+8.7pp</span>），且索引 LLM token
              仅为其 <span className="font-semibold text-ds-gray-1000">1/11.6</span>
              、查询延迟仅为其 <span className="font-semibold text-ds-gray-1000">1/5</span>
              ；在此语料形态下 LightRAG 甚至不敌 naive 向量 RAG（
              <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
                §6 外部基线
              </a>
              ）。
            </>
          }
        />
        <Contribution
          tag="F4"
          en={
            <>
              Scale is the moat — growing the corpus to 2,521 tickets collapses naive vector Hit@1
              0.738→0.242 (−49.6pp) while this system decays 0.772→
              <span className="font-semibold text-ds-gray-1000">0.490</span> (
              <span className="font-semibold text-ds-gray-1000">−28.2pp</span>, about half the
              slope) and leads every metric at production scale (Hit@1{' '}
              <span className="font-semibold text-ds-gray-1000">+24.8pp</span>); the 600-ticket
              rerun confirms the lead widens as the corpus grows (
              <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
                §6 external baselines
              </a>
              ).
            </>
          }
          zh={
            <>
              规模是护城河：语料扩到 2521 张工单时，naive 向量的 Hit@1 从 0.738 崩至
              0.242（−49.6pp），本系统仅从 0.772 降至{' '}
              <span className="font-semibold text-ds-gray-1000">0.490</span>（
              <span className="font-semibold text-ds-gray-1000">−28.2pp</span>
              ，斜率约减半），且在生产规模下全部指标领先（Hit@1{' '}
              <span className="font-semibold text-ds-gray-1000">+24.8pp</span>
              ）；600 工单复跑证实领先随语料扩大（
              <a href="#s6-ext" className="text-ds-blue-900 hover:underline">
                §6 外部基线
              </a>
              ）。
            </>
          }
        />
      </div>
    </Section>
  )
}
