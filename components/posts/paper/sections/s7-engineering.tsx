import { C, P, Section } from '../shared'

export default function S7Engineering() {
  return (
    <Section id="s7" num="07" en="Engineering" zh="工程实现">
      <P
        en={
          <>
            The deployment topology has four parts: a Next.js frontend and proxy on Vercel, a
            FastAPI backend on GCP Cloud Run, Neo4j AuraDB, and a Cloudflare D1 call log. gr
            co-citation issues a single Cypher query per request (<C>GRAPH_COCITATION_QUERY</C>);
            candidate-level collect and rarity aggregation are both pushed down into Cypher to bound
            the neighborhood fan-out. Link degree is precomputed at load time (
            <C>refresh_link_degrees</C>), while Keyword degree is counted inline at query time.
            Vector similarity on the constrained path is computed on the Neo4j side with{' '}
            <C>gds.similarity.cosine</C>, returning only the top-k.
          </>
        }
        zh={
          <>
            部署拓扑由四部分组成：Vercel 上的 Next.js 前端与代理，GCP Cloud Run 上的 FastAPI
            后端，Neo4j AuraDB，以及 Cloudflare D1 调用日志。gr 共引在每次查询中只发一条 Cypher（
            <C>GRAPH_COCITATION_QUERY</C>），候选级的 collect 与 rarity 聚合都下推到
            Cypher，以控制邻域扇出。Link 的 degree 在装载期预计算（<C>refresh_link_degrees</C>
            ），Keyword 的 degree 在检索期内联计数。约束路径的向量相似度用{' '}
            <C>gds.similarity.cosine</C> 在 Neo4j 端计算，只回传 top-k。
          </>
        }
      />
      <P
        en={
          <>
            The latency budget targets online serving: warm p50 below 2s, p95 below 4s; current warm
            caches run about 520ms (local) and 250ms (same-region). Observability runs through the
            Vercel proxy: <C>after()</C> writes asynchronously to D1 <C>retrieval_calls</C>, with
            the <C>X-Query-Confidence</C> and <C>X-Missed-Retrieval</C> response headers persisted
            alongside, and <C>/dashboard</C> provides an operator view. A/B and rollback cost is
            low: both the gr and kw channels are controlled by environment-variable flags, and with
            a flag off the results are bit-identical, so a rollback is available at any time.
          </>
        }
        zh={
          <>
            延迟预算面向线上服务：warm p50 目标小于 2s，p95 目标小于 4s；当前暖缓存约为
            520ms（本地）与 250ms（同区）。可观测性由 Vercel 代理完成，<C>after()</C> 异步写入 D1{' '}
            <C>retrieval_calls</C>，<C>X-Query-Confidence</C> 与 <C>X-Missed-Retrieval</C>{' '}
            响应头一并落库，<C>/dashboard</C> 提供 operator 视图。A/B 与回滚成本很低：gr 与 kw
            通道都由环境变量开关控制，flag 关闭时结果比特级不变，可以随时回退。
          </>
        }
      />
    </Section>
  )
}
