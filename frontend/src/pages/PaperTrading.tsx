import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, RefreshCw, ShieldAlert, Target, TrendingDown, TrendingUp } from 'lucide-react'
import { fetchAPI } from '@panwatch/api/client'
import { Button } from '@panwatch/base-ui/components/ui/button'

interface PaperAccount {
  id: number
  week_key: string
  initial_capital: number
  realized_pnl: number
  current_equity: number
  peak_equity: number
  max_drawdown_pct: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  status: string
  started_at: string | null
  ended_at: string | null
}

interface PaperPosition {
  id: number
  account_id: number
  setup_key: string
  side: 'long' | 'short'
  quantity_oz: number
  entry_price: number
  stop_loss: number
  target_price: number
  current_price: number
  unrealized_pnl: number
  mfe_usd: number
  mae_usd: number
  risk_usd: number
  setup_state: string
  macro_relation: string
  price_source: string
  status: string
  opened_at: string | null
  closed_at: string | null
}

interface PaperTrade {
  id: number
  side: 'long' | 'short'
  quantity_oz: number
  entry_price: number
  exit_price: number
  stop_loss: number
  target_price: number
  pnl: number
  pnl_pct_equity: number
  r_multiple: number
  mfe_usd: number
  mae_usd: number
  risk_usd: number
  exit_reason: string
  setup_state: string
  macro_relation: string
  price_source: string
  opened_at: string | null
  closed_at: string | null
}

interface PaperSignal {
  id: number
  setup_key: string
  candidate: string
  fusion_state: string
  macro_relation: string
  event_risk: boolean
  price: number | null
  accepted: boolean
  rejection_reason: string
  observed_at: string | null
}

interface PaperSettings {
  enabled: boolean
  initial_capital: number
  risk_pct: number
  reward_risk: number
  max_leverage: number
  max_spread_bps: number
  max_hold_minutes: number
  scan_seconds: number
  timezone: string
  entry_states: string[]
  execution_allowed: boolean
  storage: string
  storage_persistent: boolean
}

interface PaperPerformance {
  trade_count: number
  average_r: number
  expectancy_r: number
  profit_factor: number | null
  average_mfe_usd: number
  average_mae_usd: number
  average_win_r: number
  average_loss_r: number
}

interface PaperWeekHistory extends PaperAccount {
  return_pct: number
  win_rate: number
  performance: PaperPerformance
}

interface PaperWeeksResponse {
  weeks: PaperWeekHistory[]
  execution_allowed: boolean
}

interface PaperEligibility {
  eligible: boolean
  gate_reason: string
  candidate: string
  fusion_state: string
  macro_relation: string
  macro_bias: number
  macro_confidence: number | null
  event_risk: boolean
  event_kind: string | null
  event_name: string | null
  event_time_utc: string | null
  event_age_minutes: number | null
  event_confidence: number | null
  event_validation: string | null
  alignment: string
  micro_direction: string
  spot: {
    price: number | null
    bid: number | null
    ask: number | null
    spread_bps: number | null
    source: string | null
    age_seconds: number | null
    is_stale: boolean | null
  }
  projected: {
    side: 'long' | 'short'
    entry_price: number
    stop_loss: number
    target_price: number
    quantity_oz: number
    risk_usd: number
  } | null
  position: PaperPosition | null
  execution_allowed: boolean
}

interface PaperSummary {
  account: PaperAccount | null
  position: PaperPosition | null
  trades: PaperTrade[]
  signals: PaperSignal[]
  settings: PaperSettings
  performance: PaperPerformance
  execution_allowed: boolean
}

function money(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value)
}

function num(value?: number | null, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return value.toFixed(digits)
}

function human(value?: string | null): string {
  if (!value) return '--'
  return value.replace(/_/g, ' ').replace(/^./, (c: string) => c.toUpperCase())
}

function EquityCurve({
  initialCapital,
  currentEquity,
  trades,
}: {
  initialCapital: number
  currentEquity: number
  trades: PaperTrade[]
}) {
  const ordered = [...trades].sort((a, b) => {
    const left = a.closed_at ? new Date(a.closed_at).getTime() : 0
    const right = b.closed_at ? new Date(b.closed_at).getTime() : 0
    return left - right
  })

  const values = [initialCapital]
  let equity = initialCapital
  for (const trade of ordered) {
    equity += trade.pnl
    values.push(equity)
  }
  const latestClosedEquity = values[values.length - 1] || initialCapital
  if (Math.abs(latestClosedEquity - currentEquity) > 0.005) {
    values.push(currentEquity)
  }

  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  const span = Math.max(1, maxValue - minValue)
  const pad = Math.max(span * 0.15, initialCapital * 0.001)
  const low = minValue - pad
  const high = maxValue + pad
  const range = Math.max(1, high - low)
  const denominator = Math.max(1, values.length - 1)
  const points = values
    .map((value, index) => {
      const x = (index / denominator) * 100
      const y = 34 - ((value - low) / range) * 30
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')

  return (
    <div>
      <div className="mb-2 flex items-end justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Equity curve</div>
          <div className="mt-1 text-[12px] text-muted-foreground">
            {ordered.length} closed trade{ordered.length === 1 ? '' : 's'} · live equity included
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[16px] font-semibold">{money(currentEquity)}</div>
          <div className="text-[10px] text-muted-foreground">
            Range {money(minValue)} → {money(maxValue)}
          </div>
        </div>
      </div>
      <div className="h-40 w-full rounded-xl bg-accent/25 p-2">
        <svg viewBox="0 0 100 36" className="h-full w-full overflow-visible" role="img" aria-label="Paper trading equity curve">
          <line x1="0" x2="100" y1="34" y2="34" className="stroke-border" strokeWidth="0.35" />
          <polyline
            points={points}
            fill="none"
            vectorEffect="non-scaling-stroke"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-primary"
          />
          {values.map((value, index) => {
            const x = (index / denominator) * 100
            const y = 34 - ((value - low) / range) * 30
            return (
              <circle
                key={`${index}-${value}`}
                cx={x}
                cy={y}
                r="0.75"
                fill="currentColor"
                className="text-primary"
              />
            )
          })}
        </svg>
      </div>
    </div>
  )
}

function dateTime(value?: string | null): string {
  if (!value) return '--'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return value
  return d.toLocaleString()
}

export default function PaperTradingPage() {
  const [data, setData] = useState<PaperSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState('')
  const [weeks, setWeeks] = useState<PaperWeekHistory[]>([])
  const [eligibility, setEligibility] = useState<PaperEligibility | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [result, history] = await Promise.all([
        fetchAPI<PaperSummary>('/xau/paper/summary?trade_limit=50&signal_limit=50', {
          timeoutMs: 30000,
        }),
        fetchAPI<PaperWeeksResponse>('/xau/paper/weeks?limit=12', {
          timeoutMs: 30000,
        }),
      ])
      setData(result)
      setWeeks(history.weeks || [])

      try {
        const liveEligibility = await fetchAPI<PaperEligibility>('/xau/paper/eligibility', {
          timeoutMs: 30000,
        })
        setEligibility(liveEligibility)
      } catch {
        setEligibility(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Paper league unavailable')
    } finally {
      setLoading(false)
    }
  }, [])

  const scan = useCallback(async () => {
    setScanning(true)
    setError('')
    try {
      await fetchAPI('/xau/paper/scan', {
        method: 'POST',
        timeoutMs: 120000,
      })
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Paper scan failed')
    } finally {
      setScanning(false)
    }
  }, [load])

  useEffect(() => {
    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 60_000)
    return () => window.clearInterval(timer)
  }, [load])

  const account = data?.account
  const position = data?.position
  const winRate = useMemo(() => {
    if (!account?.total_trades) return 0
    return (account.winning_trades / account.total_trades) * 100
  }, [account])

  const returnPct = useMemo(() => {
    if (!account?.initial_capital) return 0
    return ((account.current_equity - account.initial_capital) / account.initial_capital) * 100
  }, [account])

  return (
    <div className="page-container pb-10">
      <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <span className="rounded-full bg-primary/10 px-2.5 py-1 text-[11px] font-semibold tracking-[0.14em] text-primary">
              XAU PAPER LEAGUE
            </span>
            <span className="rounded-full bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-500">
              WEEKLY $10K
            </span>
            <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[10px] font-medium text-muted-foreground">
              AUTO · 60s
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight md:text-3xl">Gold Paper Trading</h1>
          <p className="mt-1 max-w-3xl text-[13px] leading-6 text-muted-foreground">
            A separate weekly XAU/USD simulation account. It uses conservative indicative bid/ask fills,
            explicit risk rules, and never routes a live order.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => void load()} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={() => void scan()} disabled={scanning}>
            <Activity className={`mr-2 h-4 w-4 ${scanning ? 'animate-pulse' : ''}`} />
            Run paper scan
          </Button>
        </div>
      </div>

      <div className="mb-4 rounded-2xl border border-emerald-500/20 bg-emerald-500/8 p-4">
        <div className="flex items-start gap-3">
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-emerald-500" />
          <div>
            <div className="text-[13px] font-semibold">Simulation boundary is enforced</div>
            <div className="mt-1 text-[12px] leading-5 text-muted-foreground">
              Paper fills may use indicative XAU/USD bid/ask data. execution_allowed=false is hard-coded for this
              engine. A broker execution adapter is not connected.
            </div>
          </div>
        </div>
      </div>

      {data && !data.settings.storage_persistent && (
        <div className="mb-4 rounded-2xl border border-amber-500/25 bg-amber-500/8 p-4">
          <div className="text-[13px] font-semibold text-amber-500">Storage is temporary</div>
          <div className="mt-1 text-[12px] leading-5 text-muted-foreground">
            The engine is running, but weekly history is currently on local SQLite and may reset after a deployment.
            Durable Neon storage becomes active automatically when XAU_PAPER_DATABASE_URL is configured.
          </div>
        </div>
      )}

      {error && (
        <div className="mb-4 rounded-2xl border border-rose-500/20 bg-rose-500/8 p-4 text-[13px] text-rose-500">
          {error}
        </div>
      )}

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-8">
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Week</div>
          <div className="mt-2 text-[18px] font-bold">{account?.week_key || '--'}</div>
          <div className="mt-1 text-[10px] text-muted-foreground">
            {data?.settings?.timezone || 'Asia/Baghdad'} · {data?.settings?.storage_persistent ? 'NEON PERSISTENT' : 'LOCAL FALLBACK'}
          </div>
        </div>
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Equity</div>
          <div className="mt-2 font-mono text-[20px] font-bold">{money(account?.current_equity)}</div>
          <div className={`mt-1 text-[11px] ${returnPct >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
            {returnPct >= 0 ? '+' : ''}{num(returnPct, 2)}%
          </div>
        </div>
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Realized P/L</div>
          <div className={`mt-2 font-mono text-[20px] font-bold ${(account?.realized_pnl || 0) >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
            {money(account?.realized_pnl)}
          </div>
          <div className="mt-1 text-[10px] text-muted-foreground">{account?.total_trades || 0} closed trades</div>
        </div>
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Win rate</div>
          <div className="mt-2 text-[20px] font-bold">{num(winRate, 1)}%</div>
          <div className="mt-1 text-[10px] text-muted-foreground">
            {account?.winning_trades || 0}W · {account?.losing_trades || 0}L
          </div>
        </div>
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Max drawdown</div>
          <div className="mt-2 text-[20px] font-bold text-amber-500">{num(account?.max_drawdown_pct, 2)}%</div>
          <div className="mt-1 text-[10px] text-muted-foreground">Peak {money(account?.peak_equity)}</div>
        </div>
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Risk model</div>
          <div className="mt-2 text-[18px] font-bold">{num((data?.settings?.risk_pct || 0) * 100, 1)}% / trade</div>
          <div className="mt-1 text-[10px] text-muted-foreground">
            {num(data?.settings?.reward_risk, 1)}R · {num(data?.settings?.max_leverage, 1)}× · ≤{num(data?.settings?.max_spread_bps, 1)} bps · {num(data?.settings?.max_hold_minutes, 0)}m
          </div>
        </div>
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Expectancy</div>
          <div className={`mt-2 text-[20px] font-bold ${(data?.performance?.expectancy_r || 0) >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
            {num(data?.performance?.expectancy_r, 2)}R
          </div>
          <div className="mt-1 text-[10px] text-muted-foreground">
            Avg win {num(data?.performance?.average_win_r, 2)}R · avg loss {num(data?.performance?.average_loss_r, 2)}R
          </div>
        </div>
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Profit factor</div>
          <div className="mt-2 text-[20px] font-bold">
            {data?.performance?.profit_factor == null ? '--' : num(data.performance.profit_factor, 2)}
          </div>
          <div className="mt-1 text-[10px] text-muted-foreground">
            Avg MFE {money(data?.performance?.average_mfe_usd)} · MAE {money(data?.performance?.average_mae_usd)}
          </div>
        </div>
      </div>

      <div className="card mb-4 p-4">
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-[14px] font-semibold">Live entry eligibility</h2>
              <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${
                eligibility == null
                  ? 'bg-accent/60 text-muted-foreground'
                  : eligibility.eligible
                    ? 'bg-emerald-500/10 text-emerald-500'
                    : 'bg-amber-500/10 text-amber-500'
              }`}>
                {eligibility == null ? 'UNAVAILABLE' : eligibility.eligible ? 'ELIGIBLE' : 'BLOCKED'}
              </span>
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[10px] text-muted-foreground">
                PAPER ONLY
              </span>
            </div>
            <div className="mt-1 text-[12px] text-muted-foreground">
              Exact same gates used by the paper engine. No separate UI scoring.
            </div>
          </div>
          <div className="text-left md:text-right">
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Gate reason</div>
            <div className={`mt-1 text-[13px] font-semibold ${eligibility?.eligible ? 'text-emerald-500' : 'text-amber-500'}`}>
              {eligibility == null ? 'Live check unavailable' : human(eligibility.gate_reason)}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-8">
          <div className="rounded-xl bg-accent/30 p-3">
            <div className="text-[10px] text-muted-foreground">Candidate</div>
            <div className="mt-1 text-[12px] font-semibold">{human(eligibility?.candidate)}</div>
          </div>
          <div className="rounded-xl bg-accent/30 p-3">
            <div className="text-[10px] text-muted-foreground">Fusion</div>
            <div className="mt-1 text-[12px] font-semibold">{human(eligibility?.fusion_state)}</div>
          </div>
          <div className="rounded-xl bg-accent/30 p-3">
            <div className="text-[10px] text-muted-foreground">Alignment</div>
            <div className="mt-1 text-[12px] font-semibold">{human(eligibility?.alignment)}</div>
          </div>
          <div className="rounded-xl bg-accent/30 p-3">
            <div className="text-[10px] text-muted-foreground">Micro</div>
            <div className="mt-1 text-[12px] font-semibold">{human(eligibility?.micro_direction)}</div>
          </div>
          <div className="rounded-xl bg-accent/30 p-3">
            <div className="text-[10px] text-muted-foreground">Macro</div>
            <div className="mt-1 text-[12px] font-semibold">{human(eligibility?.macro_relation)}</div>
            <div className="mt-0.5 text-[9px] text-muted-foreground">
              confidence {num(eligibility?.macro_confidence, 2)}
            </div>
          </div>
          <div className="rounded-xl bg-accent/30 p-3">
            <div className="text-[10px] text-muted-foreground">Bid / Ask</div>
            <div className="mt-1 font-mono text-[12px]">
              {num(eligibility?.spot?.bid)} / {num(eligibility?.spot?.ask)}
            </div>
          </div>
          <div className="rounded-xl bg-accent/30 p-3">
            <div className="text-[10px] text-muted-foreground">Spread</div>
            <div className="mt-1 font-mono text-[12px]">{num(eligibility?.spot?.spread_bps, 2)} bps</div>
            <div className="mt-0.5 text-[9px] text-muted-foreground">
              limit ≤{num(data?.settings?.max_spread_bps, 1)}
            </div>
          </div>
          <div className="rounded-xl bg-accent/30 p-3">
            <div className="text-[10px] text-muted-foreground">Spot</div>
            <div className="mt-1 font-mono text-[12px]">{num(eligibility?.spot?.price)}</div>
            <div className="mt-0.5 truncate text-[9px] text-muted-foreground">
              {eligibility?.spot?.source || '--'}
            </div>
          </div>
        </div>

        {eligibility?.projected && (
          <div className="mt-3 grid grid-cols-2 gap-3 rounded-xl border border-emerald-500/15 bg-emerald-500/5 p-3 md:grid-cols-6">
            <div>
              <div className="text-[10px] text-muted-foreground">Projected side</div>
              <div className={`mt-1 text-[12px] font-semibold uppercase ${eligibility.projected.side === 'long' ? 'text-emerald-500' : 'text-rose-500'}`}>
                {eligibility.projected.side}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground">Entry</div>
              <div className="mt-1 font-mono text-[12px]">{num(eligibility.projected.entry_price)}</div>
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground">Stop</div>
              <div className="mt-1 font-mono text-[12px] text-rose-500">{num(eligibility.projected.stop_loss)}</div>
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground">Target</div>
              <div className="mt-1 font-mono text-[12px] text-emerald-500">{num(eligibility.projected.target_price)}</div>
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground">Size</div>
              <div className="mt-1 font-mono text-[12px]">{num(eligibility.projected.quantity_oz, 4)} oz</div>
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground">Risk</div>
              <div className="mt-1 font-mono text-[12px]">{money(eligibility.projected.risk_usd)}</div>
            </div>
          </div>
        )}

        <div className="mt-3 text-[10px] text-muted-foreground">
          Event risk: {eligibility?.event_risk ? 'ACTIVE' : 'clear'} · validation {human(eligibility?.event_validation)}
          {eligibility?.event_name ? ` · ${eligibility.event_name}` : ''}
          {eligibility?.event_time_utc ? ` · ${dateTime(eligibility.event_time_utc)}` : ''}
          {eligibility?.event_kind === 'breaking' && eligibility?.event_age_minutes != null ? ` · age ${num(eligibility.event_age_minutes, 0)}m` : ''}
          {' · '}quote age {num(eligibility?.spot?.age_seconds, 0)}s · live execution remains locked.
        </div>
      </div>

      <div className="card mb-4 p-4">
        <EquityCurve
          initialCapital={account?.initial_capital || data?.settings?.initial_capital || 10_000}
          currentEquity={account?.current_equity || data?.settings?.initial_capital || 10_000}
          trades={data?.trades || []}
        />
      </div>

      <div className="mb-4 grid grid-cols-1 gap-3 lg:grid-cols-12">
        <div className="card p-4 lg:col-span-5">
          <div className="mb-4 flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            <h2 className="text-[14px] font-semibold">Open position</h2>
          </div>

          {!position ? (
            <div className="rounded-xl bg-accent/30 p-5 text-center text-[12px] text-muted-foreground">
              Flat. The engine waits for an eligible setup.
            </div>
          ) : (
            <div>
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <div className={`flex items-center gap-2 text-[18px] font-bold uppercase ${position.side === 'long' ? 'text-emerald-500' : 'text-rose-500'}`}>
                    {position.side === 'long' ? <TrendingUp className="h-5 w-5" /> : <TrendingDown className="h-5 w-5" />}
                    {position.side}
                  </div>
                  <div className="mt-1 text-[11px] text-muted-foreground">{human(position.setup_state)}</div>
                </div>
                <div className={`font-mono text-[20px] font-bold ${position.unrealized_pnl >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                  {money(position.unrealized_pnl)}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-[12px]">
                <div className="rounded-xl bg-accent/30 p-3">
                  <div className="text-muted-foreground">Entry</div>
                  <div className="mt-1 font-mono">{num(position.entry_price)}</div>
                </div>
                <div className="rounded-xl bg-accent/30 p-3">
                  <div className="text-muted-foreground">Mark</div>
                  <div className="mt-1 font-mono">{num(position.current_price)}</div>
                </div>
                <div className="rounded-xl bg-accent/30 p-3">
                  <div className="text-muted-foreground">Stop</div>
                  <div className="mt-1 font-mono text-rose-500">{num(position.stop_loss)}</div>
                </div>
                <div className="rounded-xl bg-accent/30 p-3">
                  <div className="text-muted-foreground">Target</div>
                  <div className="mt-1 font-mono text-emerald-500">{num(position.target_price)}</div>
                </div>
                <div className="rounded-xl bg-accent/30 p-3">
                  <div className="text-muted-foreground">Size</div>
                  <div className="mt-1 font-mono">{num(position.quantity_oz, 4)} oz</div>
                </div>
                <div className="rounded-xl bg-accent/30 p-3">
                  <div className="text-muted-foreground">Initial risk</div>
                  <div className="mt-1 font-mono">{money(position.risk_usd)}</div>
                </div>
                <div className="rounded-xl bg-accent/30 p-3">
                  <div className="text-muted-foreground">MFE</div>
                  <div className="mt-1 font-mono text-emerald-500">{money(position.mfe_usd)}</div>
                </div>
                <div className="rounded-xl bg-accent/30 p-3">
                  <div className="text-muted-foreground">MAE</div>
                  <div className="mt-1 font-mono text-rose-500">{money(position.mae_usd)}</div>
                </div>
              </div>

              <div className="mt-3 text-[10px] text-muted-foreground">
                {position.price_source} · opened {dateTime(position.opened_at)}
              </div>
            </div>
          )}
        </div>

        <div className="card p-4 lg:col-span-7">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-[14px] font-semibold">Recent closed trades</h2>
            <span className="text-[10px] text-muted-foreground">{data?.trades?.length || 0} shown</span>
          </div>
          {!data?.trades?.length ? (
            <div className="rounded-xl bg-accent/30 p-5 text-center text-[12px] text-muted-foreground">
              No closed XAU paper trades this week yet.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-[11px]">
                <thead className="text-left text-muted-foreground">
                  <tr className="border-b border-border/60">
                    <th className="pb-2 pr-3">Side</th>
                    <th className="pb-2 pr-3">Entry → Exit</th>
                    <th className="pb-2 pr-3">P/L</th>
                    <th className="pb-2 pr-3">R</th>
                    <th className="pb-2 pr-3">MFE / MAE</th>
                    <th className="pb-2 pr-3">Reason</th>
                    <th className="pb-2">Closed</th>
                  </tr>
                </thead>
                <tbody>
                  {data.trades.map((trade) => (
                    <tr key={trade.id} className="border-b border-border/40">
                      <td className={`py-3 pr-3 font-semibold uppercase ${trade.side === 'long' ? 'text-emerald-500' : 'text-rose-500'}`}>
                        {trade.side}
                      </td>
                      <td className="py-3 pr-3 font-mono">{num(trade.entry_price)} → {num(trade.exit_price)}</td>
                      <td className={`py-3 pr-3 font-mono font-semibold ${trade.pnl >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                        {money(trade.pnl)}
                      </td>
                      <td className="py-3 pr-3 font-mono">{num(trade.r_multiple, 2)}R</td>
                      <td className="py-3 pr-3 font-mono">
                        <span className="text-emerald-500">{money(trade.mfe_usd)}</span>
                        {' / '}
                        <span className="text-rose-500">{money(trade.mae_usd)}</span>
                      </td>
                      <td className="py-3 pr-3">{human(trade.exit_reason)}</td>
                      <td className="py-3 text-muted-foreground">{dateTime(trade.closed_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <div className="card mb-4 p-4">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-[14px] font-semibold">Weekly history</h2>
          <span className="text-[10px] text-muted-foreground">Last {weeks.length || 0} league weeks</span>
        </div>
        {!weeks.length ? (
          <div className="rounded-xl bg-accent/30 p-5 text-center text-[12px] text-muted-foreground">
            Weekly history will build automatically as each Baghdad-time week closes.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] text-[11px]">
              <thead className="text-left text-muted-foreground">
                <tr className="border-b border-border/60">
                  <th className="pb-2 pr-3">Week</th>
                  <th className="pb-2 pr-3">Equity</th>
                  <th className="pb-2 pr-3">Return</th>
                  <th className="pb-2 pr-3">Trades</th>
                  <th className="pb-2 pr-3">Win rate</th>
                  <th className="pb-2 pr-3">Expectancy</th>
                  <th className="pb-2 pr-3">Profit factor</th>
                  <th className="pb-2">Max DD</th>
                </tr>
              </thead>
              <tbody>
                {weeks.map((week) => (
                  <tr key={week.id} className="border-b border-border/40">
                    <td className="py-3 pr-3 font-semibold">{week.week_key}</td>
                    <td className="py-3 pr-3 font-mono">{money(week.current_equity)}</td>
                    <td className={`py-3 pr-3 font-mono font-semibold ${week.return_pct >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                      {week.return_pct >= 0 ? '+' : ''}{num(week.return_pct, 2)}%
                    </td>
                    <td className="py-3 pr-3">{week.total_trades}</td>
                    <td className="py-3 pr-3">{num(week.win_rate, 1)}%</td>
                    <td className={`py-3 pr-3 font-mono ${week.performance.expectancy_r >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                      {num(week.performance.expectancy_r, 2)}R
                    </td>
                    <td className="py-3 pr-3 font-mono">
                      {week.performance.profit_factor == null ? '--' : num(week.performance.profit_factor, 2)}
                    </td>
                    <td className="py-3 font-mono text-amber-500">{num(week.max_drawdown_pct, 2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card p-4">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-[14px] font-semibold">Setup audit trail</h2>
          <span className="text-[10px] text-muted-foreground">
            Every deduplicated long/short setup is recorded
          </span>
        </div>
        {!data?.signals?.length ? (
          <div className="rounded-xl bg-accent/30 p-5 text-center text-[12px] text-muted-foreground">
            No directional setup has been recorded this week yet.
          </div>
        ) : (
          <div className="space-y-2">
            {data.signals.map((signal) => (
              <div key={signal.id} className="flex flex-col gap-2 rounded-xl bg-accent/30 p-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`text-[12px] font-semibold uppercase ${signal.candidate === 'long_setup' ? 'text-emerald-500' : 'text-rose-500'}`}>
                      {human(signal.candidate)}
                    </span>
                    <span className="rounded-full bg-background/70 px-2 py-0.5 text-[10px]">{human(signal.fusion_state)}</span>
                    <span className="rounded-full bg-background/70 px-2 py-0.5 text-[10px]">macro {signal.macro_relation}</span>
                  </div>
                  <div className="mt-1 text-[10px] text-muted-foreground">
                    {dateTime(signal.observed_at)} · spot {num(signal.price)}
                  </div>
                </div>
                <div className={`text-[11px] font-semibold ${signal.accepted ? 'text-emerald-500' : 'text-amber-500'}`}>
                  {signal.accepted ? 'ACCEPTED' : `REJECTED · ${human(signal.rejection_reason)}`}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
