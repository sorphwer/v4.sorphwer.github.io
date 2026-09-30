import { useLang, useT } from '@/components/article/lang'
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
  const t = useT()
  // English "summary · vector · full text" needs a wider ticket box; keep it centred on x=250
  const boxW = useLang() === 'en' ? 156 : 108
  const [ref, revealed] = useReveal(0.4)

  return (
    <figure className={`fig${revealed ? ' in' : ''}`} id="fig-schema" ref={ref}>
      <div className="panel">
        <svg
          viewBox="0 0 660 300"
          role="img"
          aria-label={t(
            'Graph structure: tickets link to keywords, versions, links and assignees; customer org, raw tags and the like are left out',
            '图的结构：工单连向关键词、版本、链接、处理人；客户组织、原始标签等没有放进图'
          )}
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
                {t('has', '带有')}
              </text>
              <text x="168" y="214" textAnchor="middle">
                {t('affects', '影响版本')}
              </text>
              <text x="334" y="96" textAnchor="middle">
                {t('cites', '引用')}
              </text>
              <text x="334" y="214" textAnchor="middle">
                {t('assignee', '处理人')}
              </text>
            </g>
            <g className="nd" style={{ transitionDelay: '.0s' }}>
              <rect
                x={250 - boxW / 2}
                y="122"
                width={boxW}
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
                {t('Ticket', '工单')}
              </text>
              <text
                x="250"
                y="165"
                textAnchor="middle"
                fontSize="11"
                style={{ fill: 'var(--sub)' }}
              >
                {t('summary · vector · full text', '摘要 · 向量 · 全文')}
              </text>
            </g>
            <Attr
              x={96}
              y={56}
              delay=".35s"
              ink="var(--green)"
              wash="var(--green-bg)"
              label={t('Keywords', '关键词')}
            />
            <Attr
              x={96}
              y={240}
              delay=".45s"
              ink="var(--blue)"
              wash="var(--blue-bg)"
              label={t('Versions', '版本')}
            />
            <Attr
              x={404}
              y={56}
              delay=".55s"
              ink="var(--purple)"
              wash="var(--purple-bg)"
              label={t('External links', '外部链接')}
            />
            <Attr
              x={404}
              y={240}
              delay=".65s"
              ink="var(--gray)"
              wash="var(--gray-bg)"
              label={t('Support engineer', '支持工程师')}
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
                {t('Deliberately left out', '刻意没放进去的')}
              </text>
              <g fontSize="13.5" textDecoration="line-through" style={{ fill: 'var(--faint)' }}>
                <text x="500" y="84">
                  {t('Customer org', '客户组织')}
                </text>
                <text x="500" y="120">
                  {t('Raw helpdesk tags', '工单系统原始标签')}
                </text>
                <text x="500" y="156">
                  {t('Product line', '产品线')}
                </text>
                <text x="500" y="192">
                  {t('Environment notes', '环境描述')}
                </text>
                <text x="500" y="228">
                  {t('Customer contacts', '客户侧联系人')}
                </text>
              </g>
            </g>
          </g>
        </svg>
      </div>
      <figcaption className="cap">
        {t(
          'The whole graph schema. Everything on the right is in the ticket text; it just stays out of the graph.',
          '图的全部结构。右侧这些信息都在原文里，只是不进图。'
        )}
      </figcaption>
    </figure>
  )
}
