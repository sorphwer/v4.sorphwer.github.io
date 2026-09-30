/**
 * All quantitative figures of the paper, ported from the source document's
 * Chart.js blocks to recharts on the shadcn chart shell. Data values are
 * frozen copies of the source document; charts render with animation off so
 * language toggles repaint without replaying entrances.
 */

import type { ReactElement } from 'react'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from 'recharts'
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from './ui/chart'
import { useT } from '@/components/article/lang'

/* Chart roles (DESIGN.md): tints of the one theme blue for this system's variants, grey
 * for baselines, Warning Pink for the Zendesk anchor / significant regressions. CSS vars
 * so charts flip with theme. */
const OURS = 'var(--ds-blue-700)'
const MID = 'var(--ds-blue-600)'
const EXT = 'var(--ds-blue-500)'
const GRAY = 'var(--ds-gray-500)'
const WARN = 'var(--ds-red-700)'
const GRID = 'var(--ds-gray-alpha-200)'

const TALL = 'h-[340px] w-full aspect-auto'
const SHORT = 'h-[292px] w-full aspect-auto'

function ChartBox({
  short = false,
  config = {},
  children,
}: {
  short?: boolean
  config?: ChartConfig
  children: ReactElement
}) {
  return (
    <div className="rounded-md bg-ds-background-100 p-4 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-400)]">
      <ChartContainer config={config} className={short ? SHORT : TALL}>
        {children}
      </ChartContainer>
    </div>
  )
}

const AXIS_TICK = { fontSize: 11.5 } as const

/* ---------------------------------------------------------------- log bars */

function fmtMagnitude(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(v % 1_000_000 === 0 ? 0 : 1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(v % 1_000 === 0 ? 0 : 1)}K`
  return String(v)
}

/**
 * Bar chart on a log₁₀ axis. Recharts bars anchor at the axis baseline, so a
 * native log scale degenerates; plotting log₁₀(v) on a linear axis with
 * power-of-ten ticks preserves both bar geometry and the log reading.
 */
function LogBars({
  data,
  yLabel,
  unit,
  short = true,
}: {
  data: { label: string; value: number; color: string }[]
  yLabel: string
  unit: string
  short?: boolean
}) {
  const rows = data.map((d) => ({ ...d, log: Math.log10(d.value) }))
  const maxExp = Math.ceil(Math.max(...rows.map((r) => r.log)))
  const minExp = Math.floor(Math.min(...rows.map((r) => r.log))) - 1
  const ticks = Array.from({ length: maxExp - minExp + 1 }, (_, i) => minExp + i)

  return (
    <ChartBox short={short}>
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="label" tickLine={false} axisLine={false} tick={AXIS_TICK} interval={0} />
        <YAxis
          domain={[minExp, maxExp]}
          ticks={ticks}
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          tickFormatter={(v: number) => fmtMagnitude(10 ** v)}
          label={{
            value: yLabel,
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <ChartTooltip
          content={({ active, payload }) =>
            active && payload?.length ? (
              <div className="rounded-md border border-ds-gray-alpha-400 bg-ds-background-100 px-2.5 py-1.5 text-xs shadow-md">
                {payload[0].payload.label}:{' '}
                <span className="font-geist-mono font-medium">
                  {payload[0].payload.value.toLocaleString()}
                </span>{' '}
                {unit}
              </div>
            ) : null
          }
        />
        <Bar dataKey="log" radius={[3, 3, 0, 0]} barSize={64} isAnimationActive={false}>
          {rows.map((r) => (
            <Cell key={r.label} fill={r.color} />
          ))}
        </Bar>
      </BarChart>
    </ChartBox>
  )
}

/* --------------------------------------------------------------- Figure 4 */

export function FigRRF() {
  const t = useT()
  const data = Array.from({ length: 10 }, (_, i) => ({
    r: i + 1,
    vote: 1 / (60 + i + 1),
  }))
  return (
    <ChartBox>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 8 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis
          dataKey="r"
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          label={{
            value: t('in-channel rank r', '通道内名次 r'),
            position: 'insideBottom',
            offset: -2,
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <YAxis
          domain={[0, 0.018]}
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          tickFormatter={(v: number) => v.toFixed(3)}
          label={{
            value: t('RRF vote value', 'RRF 票值'),
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              hideLabel
              formatter={(value) => (
                <span className="font-geist-mono">{Number(value).toFixed(4)}</span>
              )}
            />
          }
        />
        <Line
          dataKey="vote"
          name={t('per-channel vote 1/(60+r)', '单通道贡献 1/(60+r)')}
          stroke={OURS}
          strokeWidth={2}
          dot={{ r: 3.5, fill: OURS, strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ChartBox>
  )
}

/* --------------------------------------------------------------- Figure 5 */

export function FigCocit() {
  const t = useT()
  return (
    <LogBars
      yLabel={t('ticket pairs (log axis)', '工单对数量（对数轴）')}
      unit={t('pairs', '对')}
      data={[
        { label: t('share ≥1 Link', '共享 ≥1 Link'), value: 881, color: OURS },
        { label: t('share ≥1 Keyword', '共享 ≥1 Keyword'), value: 747636, color: MID },
      ]}
    />
  )
}

/* --------------------------------------------------------------- Figure 6 */

const RARITY = [
  { deg: 2, r: 0.85 },
  { deg: 3, r: 0.82 },
  { deg: 15, r: 0.63 },
  { deg: 50, r: 0.48 },
  { deg: 120, r: 0.36 },
  { deg: 400, r: 0.2 },
]

export function FigRarity() {
  const t = useT()
  return (
    <ChartBox short>
      <BarChart
        data={RARITY.map((d) => ({ label: `deg ${d.deg}`, r: d.r }))}
        margin={{ top: 8, right: 8, bottom: 4, left: 8 }}
      >
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis
          dataKey="label"
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          interval={0}
          label={{
            value: t('feature ticket-degree', '特征的 ticket-degree'),
            position: 'insideBottom',
            offset: -2,
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <YAxis
          domain={[0, 1]}
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          label={{
            value: t('rarity r(u) ∈ [0,1]', '稀有度 r(u) ∈ [0,1]'),
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              hideLabel
              formatter={(value) => (
                <span className="font-geist-mono">r(u) = {Number(value).toFixed(2)}</span>
              )}
            />
          }
        />
        <Bar dataKey="r" fill={OURS} radius={[3, 3, 0, 0]} barSize={40} isAnimationActive={false} />
      </BarChart>
    </ChartBox>
  )
}

/* --------------------------------------------------------------- Figure 7 */

export function FigBound() {
  const t = useT()
  const rows = [
    { label: t('gr-only ceiling', 'gr-only 新票上界'), v: 0.0115, color: OURS },
    { label: t('1-channel rank 1', '单通道 rank1'), v: 0.0164, color: GRAY },
    { label: t('2-channel rank 1', '双通道 rank1'), v: 0.0328, color: GRAY },
    { label: t('3-channel rank 1', '三通道 rank1'), v: 0.0492, color: GRAY },
  ]
  return (
    <ChartBox short>
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="label" tickLine={false} axisLine={false} tick={AXIS_TICK} interval={0} />
        <YAxis
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          tickFormatter={(v: number) => v.toFixed(3)}
          label={{
            value: t('fused RRF score', 'RRF 融合分'),
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              hideLabel
              formatter={(value) => (
                <span className="font-geist-mono">{Number(value).toFixed(4)}</span>
              )}
            />
          }
        />
        <Bar dataKey="v" radius={[3, 3, 0, 0]} barSize={48} isAnimationActive={false}>
          {rows.map((r) => (
            <Cell key={r.label} fill={r.color} />
          ))}
        </Bar>
      </BarChart>
    </ChartBox>
  )
}

/* ------------------------------------------------------- grouped metric bars */

function GroupedMetricBars({
  data,
  series,
  config,
}: {
  data: Record<string, string | number>[]
  series: { key: string; color: string }[]
  config: ChartConfig
}) {
  return (
    <ChartBox config={config}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="metric" tickLine={false} axisLine={false} tick={AXIS_TICK} interval={0} />
        <YAxis domain={[0, 1]} tickLine={false} axisLine={false} tick={AXIS_TICK} />
        <ChartTooltip
          content={
            <ChartTooltipContent
              formatter={(value, name, item) => (
                <>
                  <span
                    className="mt-0.5 inline-block size-2.5 shrink-0 rounded-[2px]"
                    style={{ background: item?.color as string }}
                  />
                  <span className="text-ds-muted-foreground">
                    {String(config[name as string]?.label ?? name)}
                  </span>
                  <span className="ml-auto font-geist-mono font-medium">
                    {Number(value).toFixed(3)}
                  </span>
                </>
              )}
            />
          }
        />
        <ChartLegend content={<ChartLegendContent />} />
        {series.map((s) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            fill={s.color}
            radius={[3, 3, 0, 0]}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ChartBox>
  )
}

/* --------------------------------------------------------------- Figure 9 */

export function FigMain() {
  const t = useT()
  const config: ChartConfig = {
    base: { label: 'base', color: GRAY },
    kw: { label: '+kw', color: EXT },
    gr: { label: t('+gr (kw off)', '+gr（kw 关）'), color: OURS },
    kwgr: { label: '+kw+gr', color: MID },
    zd: { label: 'zendesk', color: WARN },
  }
  const data = [
    { metric: 'Hit@1', base: 0.362, kw: 0.362, gr: 0.463, kwgr: 0.45, zd: 0.0 },
    { metric: 'Hit@5', base: 0.872, kw: 0.886, gr: 0.859, kwgr: 0.906, zd: 0.027 },
    { metric: 'MRR@10', base: 0.577, kw: 0.589, gr: 0.638, kwgr: 0.638, zd: 0.009 },
    { metric: 'nDCG@10', base: 0.634, kw: 0.644, gr: 0.671, kwgr: 0.662, zd: 0.01 },
  ]
  return (
    <GroupedMetricBars
      data={data}
      config={config}
      series={[
        { key: 'base', color: GRAY },
        { key: 'kw', color: EXT },
        { key: 'gr', color: OURS },
        { key: 'kwgr', color: MID },
        { key: 'zd', color: WARN },
      ]}
    />
  )
}

/* -------------------------------------------------------------- Figure 10 */

export function FigFinger() {
  const t = useT()
  const config: ChartConfig = {
    base: { label: 'base', color: GRAY },
    gr: { label: t('+gr (kw off)', '+gr（kw 关）'), color: OURS },
  }
  const data = [
    { metric: 'Hit@1', base: 0.362, gr: 0.463 },
    { metric: 'Hit@5', base: 0.872, gr: 0.859 },
    { metric: 'Hit@10', base: 0.953, gr: 0.946 },
  ]
  return (
    <GroupedMetricBars
      data={data}
      config={config}
      series={[
        { key: 'base', color: GRAY },
        { key: 'gr', color: OURS },
      ]}
    />
  )
}

/* -------------------------------------------------------------- Figure 11 */

const MCNEMAR = [
  { pair: 'base→+kw', p: '1.0', net: 0, sig: false },
  { pair: 'base→+gr', p: '.006', net: 15, sig: true },
  { pair: '+kw→+kw+gr', p: '.002', net: 13, sig: true },
  { pair: '+gr→+kw+gr', p: '.815', net: -2, sig: false },
  { pair: 'base→+kw+gr', p: '.019', net: 13, sig: true },
]

function McnemarTick({ x, y, payload }: { x?: number; y?: number; payload?: { value: string } }) {
  const row = MCNEMAR.find((m) => m.pair === payload?.value)
  return (
    <text x={x} y={y} dy={10} textAnchor="middle" fontSize={11} fill="var(--ds-gray-900)">
      <tspan x={x}>{payload?.value}</tspan>
      <tspan x={x} dy={13} fill="var(--ds-gray-700)">
        (p={row?.p})
      </tspan>
    </text>
  )
}

export function FigMcnemar() {
  const t = useT()
  return (
    <ChartBox>
      <BarChart data={MCNEMAR} margin={{ top: 8, right: 8, bottom: 18, left: 8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis
          dataKey="pair"
          tickLine={false}
          axisLine={false}
          interval={0}
          tick={<McnemarTick />}
        />
        <YAxis
          domain={[-3, 19]}
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          label={{
            value: t('net flipped queries', '净翻正查询数 net'),
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <ChartTooltip
          content={({ active, payload }) =>
            active && payload?.length ? (
              <div className="rounded-md border border-ds-gray-alpha-400 bg-ds-background-100 px-2.5 py-1.5 text-xs shadow-md">
                {payload[0].payload.pair}:{' '}
                <span className="font-geist-mono font-medium">
                  net = {payload[0].payload.net > 0 ? '+' : ''}
                  {payload[0].payload.net}
                </span>{' '}
                (p={payload[0].payload.p})
              </div>
            ) : null
          }
        />
        <Bar dataKey="net" radius={[3, 3, 0, 0]} barSize={44} isAnimationActive={false}>
          {MCNEMAR.map((m) => (
            <Cell key={m.pair} fill={m.net < 0 ? WARN : m.sig ? OURS : GRAY} />
          ))}
        </Bar>
      </BarChart>
    </ChartBox>
  )
}

/* -------------------------------------------------------------- Figure 12 */

export function FigLatency() {
  const t = useT()
  return (
    <LogBars
      yLabel={t('p50 latency / ms (log axis)', 'p50 延迟 / ms（对数轴）')}
      unit="ms"
      data={[
        { label: 'base', value: 859, color: GRAY },
        { label: '+kw', value: 726, color: EXT },
        { label: '+gr', value: 809, color: OURS },
        { label: '+kw+gr', value: 936, color: MID },
        { label: 'zendesk', value: 26432, color: WARN },
      ]}
    />
  )
}

/* -------------------------------------------------------------- Figure 13 */

export function FigExt() {
  const t = useT()
  const config: ChartConfig = {
    kg: { label: t('this system (kg-subset)', '本系统 (kg-subset)'), color: OURS },
    naive: { label: t('naive vectors', 'naive 向量'), color: GRAY },
    lightrag: { label: 'LightRAG hybrid', color: EXT },
  }
  const data = [
    { metric: 'Hit@1', kg: 0.772, naive: 0.738, lightrag: 0.685 },
    { metric: 'Hit@5', kg: 0.966, naive: 0.987, lightrag: 0.919 },
    { metric: 'Hit@10', kg: 0.966, naive: 0.993, lightrag: 0.96 },
    { metric: 'MRR@10', kg: 0.856, naive: 0.841, lightrag: 0.764 },
    { metric: 'nDCG@10', kg: 0.857, naive: 0.835, lightrag: 0.706 },
  ]
  return (
    <GroupedMetricBars
      data={data}
      config={config}
      series={[
        { key: 'kg', color: OURS },
        { key: 'naive', color: GRAY },
        { key: 'lightrag', color: EXT },
      ]}
    />
  )
}

/* -------------------------------------------------------------- Figure 14 */

export function FigExtCost() {
  const t = useT()
  return (
    <LogBars
      yLabel={t('total tokens (log axis)', 'token 总量（对数轴）')}
      unit="tokens"
      data={[
        { label: t('this system ingest', '本系统 ingest'), value: 808000, color: OURS },
        { label: t('naive vectors', 'naive 向量'), value: 574000, color: GRAY },
        { label: 'LightRAG hybrid', value: 9260000, color: EXT },
      ]}
    />
  )
}

/* -------------------------------------------------------------- Figure 15 */

const SCALE_SIZES = [300, 450, 600, 900, 1200, 1600, 2000, 2250, 2521]
const NAIVE_MEAN = [0.738, 0.651, 0.579, 0.501, 0.414, 0.327, 0.277, 0.262, 0.242]
const NAIVE_MIN = [0.738, 0.638, 0.544, 0.477, 0.403, 0.315, 0.268, 0.255, 0.242]
const NAIVE_MAX = [0.738, 0.658, 0.611, 0.537, 0.436, 0.342, 0.289, 0.268, 0.242]
const KG_POINTS: Record<number, number> = { 300: 0.772, 600: 0.711, 2521: 0.49 }
const LR_POINTS: Record<number, number> = { 300: 0.685, 600: 0.57 }

export function FigScale() {
  const t = useT()
  const data = SCALE_SIZES.map((n, i) => ({
    n,
    naive: NAIVE_MEAN[i],
    band: [NAIVE_MIN[i], NAIVE_MAX[i]],
    fit: 2.121 - 0.241 * Math.log(n),
    kg: KG_POINTS[n],
    lightrag: LR_POINTS[n],
  }))
  const config: ChartConfig = {
    naive: {
      label: t('naive vectors (9 sizes × 3 seeds, mean)', 'naive 向量（9 规模 × 3 seeds 均值）'),
      color: GRAY,
    },
    fit: {
      label: t('fit 2.121 − 0.241·ln N', '拟合 2.121 − 0.241·ln N'),
      color: 'var(--ds-gray-900)',
    },
    kg: {
      label: t('this system (3 points, dashed guide)', '本系统（三点，示意连线）'),
      color: OURS,
    },
    lightrag: {
      label: t('LightRAG hybrid (2 measured points)', 'LightRAG hybrid（两点实测）'),
      color: EXT,
    },
  }

  return (
    <ChartBox config={config}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis
          dataKey="n"
          type="number"
          scale="log"
          domain={[280, 2700]}
          ticks={[300, 450, 600, 900, 1200, 1600, 2000, 2521]}
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          label={{
            value: t('corpus size (tickets, log axis)', '语料规模（工单数，对数轴）'),
            position: 'insideBottom',
            offset: -2,
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <YAxis
          domain={[0, 1]}
          tickLine={false}
          axisLine={false}
          tick={AXIS_TICK}
          label={{
            value: 'Hit@1',
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11.5, fill: 'var(--ds-gray-900)' },
          }}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              formatter={(value, name, item) => {
                if (name === 'band') return null
                return (
                  <>
                    <span
                      className="mt-0.5 inline-block size-2.5 shrink-0 rounded-[2px]"
                      style={{ background: item?.color as string }}
                    />
                    <span className="text-ds-muted-foreground">
                      {String(config[name as string]?.label ?? name)}
                    </span>
                    <span className="ml-auto font-geist-mono font-medium">
                      {Number(value).toFixed(3)}
                    </span>
                  </>
                )
              }}
            />
          }
        />
        <ChartLegend content={<ChartLegendContent />} />
        <Area
          dataKey="band"
          stroke="none"
          fill="var(--ds-gray-alpha-200)"
          isAnimationActive={false}
          legendType="none"
          tooltipType="none"
        />
        <Line
          dataKey="naive"
          name="naive"
          stroke={GRAY}
          strokeWidth={2}
          dot={{ r: 3.5, fill: GRAY, strokeWidth: 0 }}
          isAnimationActive={false}
        />
        <Line
          dataKey="fit"
          name="fit"
          stroke="var(--ds-gray-900)"
          strokeWidth={1.5}
          strokeDasharray="3 3"
          dot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="kg"
          name="kg"
          stroke={OURS}
          strokeWidth={2}
          strokeDasharray="6 4"
          dot={{ r: 4.5, fill: OURS, strokeWidth: 0 }}
          connectNulls
          isAnimationActive={false}
        />
        <Line
          dataKey="lightrag"
          name="lightrag"
          stroke={EXT}
          strokeWidth={2}
          strokeDasharray="6 4"
          dot={{ r: 4.5, fill: EXT, strokeWidth: 0 }}
          connectNulls
          isAnimationActive={false}
        />
      </ComposedChart>
    </ChartBox>
  )
}
