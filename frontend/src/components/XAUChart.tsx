import { useMemo } from 'react'

export interface XAUChartBar {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  ema9: number | null
  ema21: number | null
  ema50: number | null
  ema200: number | null
  ema1000: number | null
  source: string
}

interface Props {
  bars: XAUChartBar[]
  loading?: boolean
  swingHigh?: number | null
  swingLow?: number | null
  triggerLevel?: number | null
}

function pathFor(values: Array<number | null>, x: (i: number) => number, y: (v: number) => number): string {
  let started = false
  return values.map((value, i) => {
    if (value == null || !Number.isFinite(value)) return ''
    const command = started ? 'L' : 'M'
    started = true
    return `${command}${x(i).toFixed(2)},${y(value).toFixed(2)}`
  }).join(' ')
}

export default function XAUChart({ bars, loading, swingHigh, swingLow, triggerLevel }: Props) {
  const model = useMemo(() => {
    const rows = bars.slice(-120)
    if (!rows.length) return null
    const levels = [swingHigh, swingLow, triggerLevel].filter((v): v is number => v != null && Number.isFinite(v))
    const lows = rows.map(r => r.low)
    const highs = rows.map(r => r.high)
    const rawMin = Math.min(...lows, ...levels)
    const rawMax = Math.max(...highs, ...levels)
    const pad = Math.max((rawMax - rawMin) * 0.08, 0.5)
    const min = rawMin - pad
    const max = rawMax + pad
    const width = 1000
    const height = 420
    const left = 18
    const right = 76
    const top = 18
    const bottom = 34
    const plotW = width - left - right
    const plotH = height - top - bottom
    const step = plotW / Math.max(1, rows.length)
    const candleW = Math.max(2.2, Math.min(7, step * 0.58))
    const x = (i: number) => left + step * i + step / 2
    const y = (v: number) => top + (max - v) / Math.max(1e-9, max - min) * plotH
    const ema9 = pathFor(rows.map(r => r.ema9), x, y)
    const ema21 = pathFor(rows.map(r => r.ema21), x, y)
    const ema50 = pathFor(rows.map(r => r.ema50), x, y)
    const ema200 = pathFor(rows.map(r => r.ema200), x, y)
    const ema1000 = pathFor(rows.map(r => r.ema1000), x, y)
    const ticks = Array.from({ length: 6 }, (_, i) => max - (max - min) * i / 5)
    return { rows, width, height, left, right, top, bottom, plotW, plotH, candleW, x, y, ema9, ema21, ema50, ema200, ema1000, ticks, min, max }
  }, [bars, swingHigh, swingLow, triggerLevel])

  if (loading && !bars.length) {
    return <div className="flex h-[360px] items-center justify-center text-xs text-muted-foreground">Loading market structure…</div>
  }
  if (!model) {
    return <div className="flex h-[360px] items-center justify-center text-xs text-muted-foreground">Chart data unavailable.</div>
  }

  const { rows, width, height, left, top, plotW, plotH, candleW, x, y, ema9, ema21, ema50, ema200, ema1000, ticks } = model
  const latest = rows[rows.length - 1]

  const renderLevel = (value: number | null | undefined, label: string, className: string) => {
    if (value == null || !Number.isFinite(value)) return null
    const yy = y(value)
    if (yy < top || yy > top + plotH) return null
    return (
      <g>
        <line x1={left} x2={left + plotW} y1={yy} y2={yy} className={className} strokeDasharray="5 5" strokeWidth="1" />
        <text x={left + plotW - 6} y={yy - 5} textAnchor="end" className="fill-muted-foreground text-[10px]">
          {label} {value.toFixed(2)}
        </text>
      </g>
    )
  }

  return (
    <div className="relative w-full overflow-hidden rounded-2xl bg-background/20">
      <svg viewBox={`0 0 ${width} ${height}`} className="block w-full min-w-[720px]" role="img" aria-label="XAU/USD candlestick chart">
        {ticks.map((tick, i) => {
          const yy = y(tick)
          return (
            <g key={i}>
              <line x1={left} x2={left + plotW} y1={yy} y2={yy} className="stroke-border/50" strokeWidth="1" />
              <text x={left + plotW + 10} y={yy + 4} className="fill-muted-foreground text-[10px]">{tick.toFixed(2)}</text>
            </g>
          )
        })}

        {renderLevel(swingHigh, 'Swing H', 'stroke-rose-500/70')}
        {renderLevel(swingLow, 'Swing L', 'stroke-emerald-500/70')}
        {renderLevel(triggerLevel, 'Trigger', 'stroke-amber-500/70')}

        {rows.map((bar, i) => {
          const xx = x(i)
          const up = bar.close >= bar.open
          const topBody = y(Math.max(bar.open, bar.close))
          const bottomBody = y(Math.min(bar.open, bar.close))
          const bodyH = Math.max(1.4, bottomBody - topBody)
          return (
            <g key={bar.time}>
              <line x1={xx} x2={xx} y1={y(bar.high)} y2={y(bar.low)} className={up ? 'stroke-emerald-500' : 'stroke-rose-500'} strokeWidth="1.2" />
              <rect x={xx - candleW / 2} y={topBody} width={candleW} height={bodyH} rx="0.7" className={up ? 'fill-emerald-500' : 'fill-rose-500'} />
            </g>
          )
        })}

        <path d={ema1000} fill="none" className="stroke-muted-foreground/45" strokeWidth="1.1" strokeDasharray="7 6" />
        <path d={ema200} fill="none" className="stroke-amber-500/70" strokeWidth="1.15" />
        <path d={ema50} fill="none" className="stroke-cyan-500/70" strokeWidth="1.2" />
        <path d={ema21} fill="none" className="stroke-violet-500" strokeWidth="1.35" />
        <path d={ema9} fill="none" className="stroke-blue-500" strokeWidth="1.45" />

        {rows.filter((_, i) => i % Math.max(1, Math.floor(rows.length / 6)) === 0).map((bar) => {
          const sourceIndex = rows.indexOf(bar)
          const dt = new Date(bar.time)
          return <text key={bar.time} x={x(sourceIndex)} y={height - 10} textAnchor="middle" className="fill-muted-foreground text-[10px]">{dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</text>
        })}

        <g>
          <rect x={left + plotW - 76} y={Math.max(top + 4, y(latest.close) - 12)} width="76" height="24" rx="6" className={latest.close >= latest.open ? 'fill-emerald-500' : 'fill-rose-500'} />
          <text x={left + plotW - 38} y={Math.max(top + 19, y(latest.close) + 4)} textAnchor="middle" className="fill-white text-[11px] font-semibold">{latest.close.toFixed(2)}</text>
        </g>
      </svg>
      <div className="absolute left-3 top-3 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg bg-background/75 px-2.5 py-1.5 text-[10px] backdrop-blur">
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-blue-500"/>9</span>
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-violet-500"/>21</span>
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-cyan-500"/>50</span>
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-500"/>200</span>
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-muted-foreground"/>1000</span>
      </div>
    </div>
  )
}
