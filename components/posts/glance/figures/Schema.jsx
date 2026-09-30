import { useReveal } from '../shared'

const EDGE = { stroke: 'var(--edge)' }

function Attr({ x, y, delay, ink, wash, label }) {
  return (
    <g className="nd" style={{ transitionDelay: delay }}>
      <circle cx={x} cy={y} r="26" style={{ fill: wash }} />
      <circle cx={x} cy={y} r="8" style={{ fill: ink }} />
      <text
        x={x}
        y={y + 48}
        textAnchor="middle"
        fontSize="13"
        fontWeight="600"
        style={{ fill: ink }}
      >
        {label}
      </text>
    </g>
  )
}

export default function Schema() {
  const [ref, revealed] = useReveal(0.4)

  return (
    <figure className={`fig${revealed ? ' in' : ''}`} id="fig-schema" ref={ref}>
      <div className="panel">
        <svg
          viewBox="0 0 660 300"
          role="img"
          aria-label="图的结构：工单连向关键词、版本、链接、处理人；客户组织、原始标签等没有放进图"
        >
          <g style={{ fontFamily: 'var(--sans)' }}>
            <path
              className="draw"
              d="M250 150 L110 64"
              strokeWidth="1.6"
              fill="none"
              style={EDGE}
            />
            <path
              className="draw"
              d="M250 150 L100 236"
              strokeWidth="1.6"
              fill="none"
              style={EDGE}
            />
            <path
              className="draw"
              d="M250 150 L396 64"
              strokeWidth="1.6"
              fill="none"
              style={EDGE}
            />
            <path
              className="draw"
              d="M250 150 L396 236"
              strokeWidth="1.6"
              fill="none"
              style={EDGE}
            />
            <g fontSize="11.5" style={{ fill: 'var(--sub)' }}>
              <text x="168" y="96" textAnchor="middle">
                带有
              </text>
              <text x="168" y="214" textAnchor="middle">
                影响版本
              </text>
              <text x="334" y="96" textAnchor="middle">
                引用
              </text>
              <text x="334" y="214" textAnchor="middle">
                处理人
              </text>
            </g>
            <g className="nd" style={{ transitionDelay: '.0s' }}>
              <rect
                x="196"
                y="122"
                width="108"
                height="56"
                rx="10"
                strokeWidth="1.6"
                style={{ fill: 'var(--card)', stroke: 'var(--ink)' }}
              />
              <text
                x="250"
                y="146"
                textAnchor="middle"
                fontSize="15"
                fontWeight="700"
                style={{ fill: 'var(--ink)' }}
              >
                工单
              </text>
              <text
                x="250"
                y="165"
                textAnchor="middle"
                fontSize="11"
                style={{ fill: 'var(--sub)' }}
              >
                摘要 · 向量 · 全文
              </text>
            </g>
            <Attr
              x={96}
              y={56}
              delay=".35s"
              ink="var(--green)"
              wash="var(--green-bg)"
              label="关键词"
            />
            <Attr
              x={96}
              y={240}
              delay=".45s"
              ink="var(--blue)"
              wash="var(--blue-bg)"
              label="版本"
            />
            <Attr
              x={404}
              y={56}
              delay=".55s"
              ink="var(--purple)"
              wash="var(--purple-bg)"
              label="外部链接"
            />
            <Attr
              x={404}
              y={240}
              delay=".65s"
              ink="var(--gray)"
              wash="var(--gray-bg)"
              label="支持工程师"
            />
            <line
              x1="474"
              y1="30"
              x2="474"
              y2="270"
              strokeWidth="1"
              style={{ stroke: 'var(--hair)' }}
            />
            <g className="nd" style={{ transitionDelay: '.9s' }}>
              <text x="500" y="44" fontSize="12.5" style={{ fill: 'var(--sub)' }}>
                刻意没放进去的
              </text>
              <g fontSize="13.5" textDecoration="line-through" style={{ fill: 'var(--faint)' }}>
                <text x="500" y="84">
                  客户组织
                </text>
                <text x="500" y="120">
                  工单系统原始标签
                </text>
                <text x="500" y="156">
                  产品线
                </text>
                <text x="500" y="192">
                  环境描述
                </text>
                <text x="500" y="228">
                  客户侧联系人
                </text>
              </g>
            </g>
          </g>
        </svg>
      </div>
      <figcaption className="cap">图的全部结构。右侧这些信息都在原文里，只是不进图。</figcaption>
    </figure>
  )
}
