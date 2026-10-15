import type { ReactNode } from 'react'
import { Section, T } from '../shared'

type Ref = {
  n: number
  /** Authors, title and venue, already formatted. */
  text: ReactNode
  url?: string
}

/** Numbered to match the <Cite n={…} /> markers in the section files. */
const REFERENCES: Ref[] = [
  {
    n: 1,
    text: (
      <>
        S. Robertson and H. Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond.{' '}
        <i>Foundations and Trends in Information Retrieval</i>, 3(4):333–389, 2009.
      </>
    ),
    url: 'https://doi.org/10.1561/1500000019',
  },
  {
    n: 2,
    text: (
      <>
        K. Spärck Jones. A Statistical Interpretation of Term Specificity and Its Application in
        Retrieval. <i>Journal of Documentation</i>, 28(1):11–21, 1972.
      </>
    ),
    url: 'https://doi.org/10.1108/eb026526',
  },
  {
    n: 3,
    text: (
      <>
        V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih. Dense
        Passage Retrieval for Open-Domain Question Answering. In <i>Proceedings of EMNLP</i>, 2020.
      </>
    ),
    url: 'https://doi.org/10.18653/v1/2020.emnlp-main.550',
  },
  {
    n: 4,
    text: (
      <>
        G. V. Cormack, C. L. A. Clarke, and S. Büttcher. Reciprocal Rank Fusion Outperforms
        Condorcet and Individual Rank Learning Methods. In <i>Proceedings of SIGIR</i>, 2009.
      </>
    ),
    url: 'https://doi.org/10.1145/1571941.1572114',
  },
  {
    n: 5,
    text: (
      <>
        D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and J. Larson. From
        Local to Global: A Graph RAG Approach to Query-Focused Summarization.{' '}
        <i>arXiv:2404.16130</i>, 2024.
      </>
    ),
    url: 'https://arxiv.org/abs/2404.16130',
  },
  {
    n: 6,
    text: (
      <>
        H. Small. Co-citation in the Scientific Literature: A New Measure of the Relationship
        Between Two Documents. <i>Journal of the American Society for Information Science</i>,
        24(4):265–269, 1973.
      </>
    ),
    url: 'https://doi.org/10.1002/asi.4630240406',
  },
  {
    n: 7,
    text: (
      <>
        K. Järvelin and J. Kekäläinen. Cumulated Gain-Based Evaluation of IR Techniques.{' '}
        <i>ACM Transactions on Information Systems</i>, 20(4):422–446, 2002.
      </>
    ),
    url: 'https://doi.org/10.1145/582415.582418',
  },
  {
    n: 8,
    text: (
      <>
        Q. McNemar. Note on the Sampling Error of the Difference Between Correlated Proportions or
        Percentages. <i>Psychometrika</i>, 12(2):153–157, 1947.
      </>
    ),
    url: 'https://doi.org/10.1007/BF02295996',
  },
  {
    n: 9,
    text: (
      <T
        en="Dify · Zendesk Knowledge-Graph Project. Internal design documentation: system architecture and knowledge-graph schema, 2026. Not public."
        zh="Dify · Zendesk 知识图谱项目。内部设计文档：系统架构与知识图谱 schema，2026。未公开。"
      />
    ),
  },
  {
    n: 10,
    text: (
      <>
        Z. Guo, L. Xia, Y. Yu, T. Ao, and C. Huang. LightRAG: Simple and Fast Retrieval-Augmented
        Generation. <i>arXiv:2410.05779</i>, 2024.
      </>
    ),
    url: 'https://arxiv.org/abs/2410.05779',
  },
  {
    n: 11,
    text: (
      <>
        Cerebras. How We Built Our Knowledge Base. <i>Cerebras Blog</i>.
      </>
    ),
    url: 'https://www.cerebras.ai/blog/how-we-built-our-knowledge-base',
  },
]

/** Back matter: the list every <Cite n={…} /> marker jumps to (#ref-N). */
export default function References() {
  return (
    <Section id="references" en="References" zh="参考文献">
      <ol className="mt-5 space-y-1 text-sm leading-6">
        {REFERENCES.map(({ n, text, url }) => (
          <li
            key={n}
            id={`ref-${n}`}
            className="-mx-2 flex scroll-mt-24 gap-3 rounded-md px-2 py-1 transition-colors target:bg-ds-blue-100"
          >
            <span className="w-7 flex-none text-right font-geist-mono text-[13px] text-ds-blue-900">
              [{n}]
            </span>
            <span className="min-w-0">
              {text}
              {url && (
                <>
                  {' '}
                  <a href={url} className="break-all text-ds-blue-900 hover:underline">
                    {url.replace(/^https:\/\//, '')}
                  </a>
                </>
              )}
            </span>
          </li>
        ))}
      </ol>
    </Section>
  )
}
