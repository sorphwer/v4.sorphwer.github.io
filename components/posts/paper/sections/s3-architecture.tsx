import {
  C,
  Fig,
  Lead,
  M,
  P,
  Section,
  T,
  TableCaption,
  TableShell,
  Td,
  Th,
  DiagramPanel,
} from '../shared'

export default function S3Architecture() {
  return (
    <Section id="s3" num="03" en="Architecture & Knowledge Graph" zh="系统架构与知识图谱">
      <P
        en={
          <>
            The system first converts Zendesk tickets into a ticket-centric Neo4j knowledge graph,
            then a five-layer hybrid retriever serves recall to the Agent, Dify External Knowledge,
            the CLI, and the Web frontend (global architecture in{' '}
            <a href="#s1" className="text-ds-blue-900 hover:underline">
              Figure 1
            </a>
            ). The ingestion paths stay separate: the Batch pipeline handles local full rebuilds,
            and the Incremental pipeline handles single-ticket top-ups over the API. This keeps
            full-corpus experiments and live top-ups from contaminating each other.
          </>
        }
        zh={
          <>
            系统先把 Zendesk 工单转成工单中心的 Neo4j 知识图谱，再由五层混合检索器向 Agent、Dify
            External Knowledge、CLI 与 Web 前端提供召回（全局架构见{' '}
            <a href="#s1" className="text-ds-blue-900 hover:underline">
              图 1
            </a>
            ）。入图路径保持分离：Batch pipeline 负责本地全量重建，Incremental pipeline 负责 API
            单票补录。这样做可以避免全量实验和线上补录互相污染。
          </>
        }
      />

      <P
        en={
          <>
            <Lead>Knowledge-graph scope.</Lead> The current graph supports retrieval scoring with
            five node classes; the graph model carries the fields directly relevant to current
            retrieval scoring, and five other object classes fall outside this scope:
          </>
        }
        zh={
          <>
            <Lead>知识图谱范围。</Lead>
            当前图谱以 5 类节点支撑检索打分；图模型承载与当前检索打分直接相关的字段，其余 5
            类对象处于这一范围之外：
          </>
        }
      />

      <div className="mt-5">
        <TableCaption
          num={1}
          en={
            <>
              Kept and excluded elements of the knowledge graph, and each node&rsquo;s role in
              retrieval.
            </>
          }
          zh={<>知识图谱的保留与排除，及各节点在检索中的角色。</>}
        />
        <TableShell>
          <thead>
            <tr>
              <Th>
                <T en="Category" zh="类别" />
              </Th>
              <Th>
                <T en="Kept · Excluded" zh="保留 / 排除" />
              </Th>
              <Th>
                <T en="Role in retrieval" zh="在检索中的角色" />
              </Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Td mono>Ticket</Td>
              <Td>
                <T en="Kept (center node)" zh="保留（中心节点）" />
              </Td>
              <Td>
                <T
                  en="The only ranked object; carries dual embeddings + full text + status/priority"
                  zh="唯一被排序的对象，携带双嵌入 + 全文 + status/priority"
                />
              </Td>
            </tr>
            <tr>
              <Td mono>Version</Td>
              <Td>
                <T en="Kept" zh="保留" />
              </Td>
              <Td>
                <T
                  en={
                    <>
                      <strong className="font-semibold">Hard constraint</strong>: text version
                      slices the subgraph
                    </>
                  }
                  zh={
                    <>
                      <strong className="font-semibold">硬约束</strong>
                      ：文本版本切子图
                    </>
                  }
                />
              </Td>
            </tr>
            <tr>
              <Td mono>Keyword</Td>
              <Td>
                <T en="Kept" zh="保留" />
              </Td>
              <Td>
                <T
                  en="Soft signal: kw channel + gr co-citation feature"
                  zh="soft 信号：kw 通道 + gr 共引特征"
                />
              </Td>
            </tr>
            <tr>
              <Td mono>Link</Td>
              <Td>
                <T en="Kept" zh="保留" />
              </Td>
              <Td>
                <T
                  en="High-precision anchor on the result side: gr co-citation feature + link consensus"
                  zh="结果侧高精度锚点：gr 共引特征 + 链接共识"
                />
              </Td>
            </tr>
            <tr>
              <Td mono>Person</Td>
              <Td>
                <T en="Kept (support agents only)" zh="保留（仅支持客服）" />
              </Td>
              <Td>
                <T en="Used for metadata filtering only" zh="仅用于元数据过滤" />
              </Td>
            </tr>
            <tr>
              <Td muted last>
                <T
                  en="Organization / raw tags / Product / Environment / customer-side Person"
                  zh="Organization / 原始 tags / Product / Environment / 客户侧 Person"
                />
              </Td>
              <Td muted last>
                <T en="Excluded" zh="排除" />
              </Td>
              <Td muted last>
                <T
                  en="The graph model focuses on fields directly relevant to the retrieval target and able to support scoring"
                  zh="图模型聚焦于与检索目标直接相关、可支持打分的字段"
                />
              </Td>
            </tr>
          </tbody>
        </TableShell>
      </div>

      <P
        en={
          <>
            <C>Ticket</C> is the only ranked center node; the four attribute-node classes all
            connect to it within one hop. Each ticket carries an issue-summary embedding{' '}
            <M t="\vec d^{\,\text{issue}}_t" />, a solution-summary embedding{' '}
            <M t="\vec d^{\,\text{sol}}_t" />, full-text-searchable text, a version set{' '}
            <M t="V_t" />, a keyword set <M t="K_t" />, and an external-link set{' '}
            <M t="\text{refs}(t)" />.
          </>
        }
        zh={
          <>
            <C>Ticket</C> 是唯一被排序的中心节点，四类属性节点都与它一跳相连。每张工单携带 issue
            摘要嵌入 <M t="\vec d^{\,\text{issue}}_t" />
            、solution 摘要嵌入 <M t="\vec d^{\,\text{sol}}_t" />
            、可全文检索文本、版本集 <M t="V_t" />
            、关键词集 <M t="K_t" />
            、外链集 <M t="\text{refs}(t)" />。
          </>
        }
      />

      <Fig
        num={2}
        en={
          <>
            The ticket-centric star schema: four edge types <C>HANDLED_BY</C> /{' '}
            <C>AFFECTS_VERSION</C> / <C>HAS_KEYWORD</C> / <C>REFERENCES</C>, all reachable within
            one hop. Adapted from the scoring-algorithm document §1.1.
          </>
        }
        zh={
          <>
            工单中心的星型 schema：四条边型 <C>HANDLED_BY</C> / <C>AFFECTS_VERSION</C> /{' '}
            <C>HAS_KEYWORD</C> / <C>REFERENCES</C>
            ，全部一跳可达。改编自打分算法文档 §1.1。
          </>
        }
      >
        <DiagramPanel>
          <div className="grid items-center gap-x-7 gap-y-4 sm:grid-cols-[minmax(240px,1fr)_minmax(280px,1.2fr)]">
            <div className="rounded-md bg-ds-background-100 px-[18px] py-4 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-500)]">
              <div className="border-b border-ds-gray-alpha-400 pb-2 font-geist-mono text-[13px] font-medium text-ds-gray-1000">
                Ticket{' '}
                <span className="font-normal text-ds-gray-700">
                  <T en="center node" zh="中心节点" />
                </span>
              </div>
              <div className="pt-2 font-geist-mono text-[11.5px] leading-loose text-ds-gray-900">
                <T en="issue summary + embedding" zh="issue 摘要 + 嵌入" />
                <br />
                <T en="solution summary + embedding" zh="solution 摘要 + 嵌入" />
                <br />
                <T en="full-text searchable text" zh="全文可检索文本" />
                <br />
                <T en="status · priority · created_at" zh="status / priority / created_at" />
              </div>
            </div>
            <div className="flex flex-col gap-2.5">
              <div className="flex items-center gap-2.5">
                <span className="h-px flex-1 bg-ds-gray-500" />
                <span className="flex-none font-geist-mono text-[10.5px] text-ds-gray-800">
                  HANDLED_BY →
                </span>
                <span className="min-w-0 flex-1 rounded-md sm:min-w-[150px] sm:flex-none bg-ds-background-100 px-3 py-[7px] text-[12.5px] shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
                  <span className="font-geist-mono font-medium">Person</span>{' '}
                  <span className="text-ds-gray-700">
                    <T en="support agents only" zh="仅支持客服" />
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="h-px flex-1 bg-ds-gray-500" />
                <span className="flex-none font-geist-mono text-[10.5px] text-ds-gray-800">
                  AFFECTS_VERSION →
                </span>
                <span className="min-w-0 flex-1 rounded-md sm:min-w-[150px] sm:flex-none bg-ds-background-100 px-3 py-[7px] text-[12.5px] shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
                  <span className="font-geist-mono font-medium">Version</span>{' '}
                  <span className="text-ds-gray-700">
                    <T en="hard constraint" zh="硬约束" />
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="h-px flex-1 bg-ds-gray-500" />
                <span className="flex-none font-geist-mono text-[10.5px] text-ds-gray-800">
                  HAS_KEYWORD →
                </span>
                <span className="min-w-0 flex-1 rounded-md sm:min-w-[150px] sm:flex-none bg-ds-background-100 px-3 py-[7px] text-[12.5px] shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
                  <span className="font-geist-mono font-medium">Keyword</span>{' '}
                  <span className="text-ds-gray-700">
                    <T en="soft signal" zh="soft 信号" />
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="h-px flex-1 bg-ds-gray-500" />
                <span className="flex-none font-geist-mono text-[10.5px] text-ds-gray-800">
                  REFERENCES →
                </span>
                <span className="min-w-0 flex-1 rounded-md sm:min-w-[150px] sm:flex-none bg-ds-background-100 px-3 py-[7px] text-[12.5px] shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
                  <span className="font-geist-mono font-medium">Link</span>{' '}
                  <span className="text-ds-gray-700">
                    <T en="degree precomputed at load" zh="degree 装载期预计算" />
                  </span>
                </span>
              </div>
            </div>
          </div>
        </DiagramPanel>
      </Fig>

      <P
        en={
          <>
            <Lead>Embedding strategy.</Lead> The embedding model is{' '}
            <C>gemini-embedding-2-preview</C> at dimension 3072; the document side uses task type{' '}
            <C>RETRIEVAL_DOCUMENT</C> and the query side uses <C>RETRIEVAL_QUERY</C>. issue and
            solution are indexed separately, forming the two semantic channels iv and sv. Summaries
            are forced to English output to stabilize the semantic space; subject and description
            keep their original text, handed to Neo4j&rsquo;s CJK full-text index to serve mixed
            Chinese-English queries. The scale numbers change over the import process: this
            paper&rsquo;s worked example uses the demonstration value <M t="N=1862" /> (scoring),
            the co-citation statistics use <M t="N=2486" /> (measured 2026-07-06), and the Golden
            Query snapshot has <C>ticket_count=1032</C> (2026-07-05). These numbers come from
            different stages, and this paper does not merge them into a single authoritative
            snapshot.
          </>
        }
        zh={
          <>
            <Lead>Embedding 策略。</Lead>嵌入模型为 <C>gemini-embedding-2-preview</C>，维度
            3072；文档端使用 task type <C>RETRIEVAL_DOCUMENT</C>，查询端使用 <C>RETRIEVAL_QUERY</C>
            。issue 与 solution 分别建索引，因此形成 iv、sv
            两条语义通道。摘要强制输出英文，用来稳定语义空间；subject 与 description 保留原文，交给
            Neo4j 的 CJK 全文索引处理中英混合查询。规模数字会随导入过程变化：本文算例取演示值{' '}
            <M t="N=1862" />
            （打分），共引统计取 <M t="N=2486" />
            （2026-07-06 实测），Golden Query snapshot 的 <C>ticket_count=1032</C>
            （2026-07-05）。这些数字来自不同阶段，本文不把它们合并成一个权威快照。
          </>
        }
      />
    </Section>
  )
}
