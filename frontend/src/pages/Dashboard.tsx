import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, AlertTriangle, Newspaper, RefreshCw, ShieldAlert, Sparkles } from 'lucide-react'
import { fetchAPI } from '@panwatch/api/client'
import { Button } from '@panwatch/base-ui/components/ui/button'

type Direction = 'bullish' | 'bearish' | 'neutral'

interface XAUFrame {
  timeframe: string
  close: number
  ema_fast: number
  ema_slow: number
  rsi14: number
  atr14: number
  atr_pct: number
  breakout: string
  direction: Direction
  recent_swing_high: number
  recent_swing_low: number
  observed_at: string
}

interface XAUSnapshot {
  instrument: string
  name: string
  research_proxy: string
  research_source: string
  research_only: boolean
  execution_feed_connected: boolean
  execution_status: string
  price: number | null
  change_pct_1m: number | null
  observed_at: string | null
  status: string
  candidate: string
  blocked: boolean
  block_reasons: string[]
  warnings: string[]
  alignment: string
  atr_reference: number | null
  swing_high_reference: number | null
  swing_low_reference: number | null
  frames: Record<string, XAUFrame>
  disclaimer: string
}

interface MacroContext {
  observed_at: string
  bias: number
  bias_label: string
  confidence: number
  event_risk: boolean
  summary: string
  drivers: string[]
  search_ok: boolean
  search_error?: string | null
  synthesis_error?: string | null
}

function fmt(value?: number | null, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return value.toFixed(digits)
}

function pct(value?: number | null, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}%`
}

function directionClass(direction?: string): string {
  if (direction === 'bullish') return 'text-emerald-500'
  if (direction === 'bearish') return 'text-rose-500'
  return 'text-muted-foreground'
}

function candidateLabel(value?: string): string {
  if (value === 'long_setup') return 'LONG SETUP'
  if (value === 'short_setup') return 'SHORT SETUP'
  return 'NO SETUP'
}

function friendlyReason(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/^./, (char: string) => char.toUpperCase())
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const [snapshot, setSnapshot] = useState<XAUSnapshot | null>(null)
  const [macro, setMacro] = useState<MacroContext | null>(null)
  const [loading, setLoading] = useState(true)
  const [macroLoading, setMacroLoading] = useState(true)
  const [error, setError] = useState('')
  const [macroError, setMacroError] = useState('')

  const loadTechnical = useCallback(async (force = false) => {
    setLoading(true)
    setError('')
    try {
      const data = await fetchAPI<XAUSnapshot>(`/xau/snapshot${force ? '?force=true' : ''}`, {
        timeoutMs: 45000,
      })
      setSnapshot(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'XAU research snapshot unavailable')
    } finally {
      setLoading(false)
    }
  }, [])

  const loadMacro = useCallback(async (force = false) => {
    setMacroLoading(true)
    setMacroError('')
    try {
      const data = await fetchAPI<MacroContext>(`/xau/macro${force ? '?force=true' : ''}`, {
        timeoutMs: 95000,
      })
      setMacro(data)
    } catch (err) {
      setMacroError(err instanceof Error ? err.message : 'Macro research unavailable')
    } finally {
      setMacroLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadTechnical()
    void loadMacro()
  }, [loadTechnical, loadMacro])

  const frames = useMemo(
    () => ['1m', '5m', '15m'].map((key) => snapshot?.frames?.[key]).filter(Boolean) as XAUFrame[],
    [snapshot],
  )

  const gateItems = useMemo(
    () => [...(snapshot?.block_reasons || []), ...(snapshot?.warnings || [])],
    [snapshot],
  )

  const macroBiasClass =
    macro?.bias === 1 ? 'text-emerald-500' : macro?.bias === -1 ? 'text-rose-500' : 'text-muted-foreground'

  return (
    <div className="page-container pb-10">
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <span className="rounded-full bg-primary/10 px-2.5 py-1 text-[11px] font-semibold tracking-[0.14em] text-primary">
              XAU TERMINAL
            </span>
            <span className="rounded-full bg-amber-500/10 px-2.5 py-1 text-[11px] font-medium text-amber-600">
              RESEARCH ONLY
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground md:text-3xl">
            Gold / U.S. Dollar
          </h1>
          <p className="mt-1 max-w-3xl text-[13px] leading-6 text-muted-foreground">
            Fast 1m / 5m / 15m technical state, macro context and external research. The current market feed is
            GC=F and is intentionally blocked from execution use.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={() => void loadMacro(true)} disabled={macroLoading}>
            <Newspaper className={`mr-2 h-4 w-4 ${macroLoading ? 'animate-pulse' : ''}`} />
            Refresh macro
          </Button>
          <Button onClick={() => void loadTechnical(true)} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh market
          </Button>
        </div>
      </div>

      <div className="mb-4 rounded-2xl border border-amber-500/25 bg-amber-500/8 p-4">
        <div className="flex items-start gap-3">
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" />
          <div>
            <div className="text-[13px] font-semibold text-foreground">Execution gate is locked</div>
            <div className="mt-1 text-[12px] leading-5 text-muted-foreground">
              A real spot XAUUSD bid/ask feed is not connected yet. Research, AI analysis and technical state are
              active, but live entry / SL / TP prices must not use the GC=F proxy.
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-2xl border border-rose-500/20 bg-rose-500/8 p-4 text-[13px] text-rose-500">
          {error}
        </div>
      )}

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">GC=F research price</div>
          <div className="mt-2 font-mono text-[24px] font-bold text-foreground">
            {fmt(snapshot?.price)}
          </div>
          <div className={`mt-1 font-mono text-[11px] ${directionClass((snapshot?.change_pct_1m || 0) >= 0 ? 'bullish' : 'bearish')}`}>
            1m {pct(snapshot?.change_pct_1m, 3)}
          </div>
        </div>

        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Candidate</div>
          <div className={`mt-2 text-[18px] font-bold ${snapshot?.candidate === 'long_setup' ? 'text-emerald-500' : snapshot?.candidate === 'short_setup' ? 'text-rose-500' : 'text-muted-foreground'}`}>
            {candidateLabel(snapshot?.candidate)}
          </div>
          <div className="mt-1 text-[11px] text-muted-foreground">
            {snapshot?.blocked ? 'Blocked by gates' : 'Research engine ready'}
          </div>
        </div>

        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Timeframe alignment</div>
          <div className={`mt-2 text-[18px] font-bold uppercase ${directionClass(snapshot?.alignment)}`}>
            {snapshot?.alignment || '--'}
          </div>
          <div className="mt-1 text-[11px] text-muted-foreground">1m · 5m · 15m</div>
        </div>

        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Macro bias</div>
          <div className={`mt-2 text-[18px] font-bold uppercase ${macroBiasClass}`}>
            {macroLoading ? 'LOADING' : macro?.bias_label || '--'}
          </div>
          <div className="mt-1 text-[11px] text-muted-foreground">
            Confidence {macro ? `${Math.round((macro.confidence || 0) * 100)}%` : '--'}
          </div>
        </div>

        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Execution feed</div>
          <div className="mt-2 text-[18px] font-bold text-amber-500">LOCKED</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Spot bid/ask not connected</div>
        </div>
      </div>

      <div className="mb-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
        {frames.map((frame) => (
          <div key={frame.timeframe} className="card p-4">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <div className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{frame.timeframe}</div>
                <div className={`mt-1 text-[18px] font-bold uppercase ${directionClass(frame.direction)}`}>
                  {frame.direction}
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-[18px] font-semibold">{fmt(frame.close)}</div>
                <div className="text-[10px] text-muted-foreground">GC=F proxy</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-x-4 gap-y-3 text-[12px]">
              <div>
                <div className="text-muted-foreground">EMA 9</div>
                <div className="mt-0.5 font-mono">{fmt(frame.ema_fast)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">EMA 21</div>
                <div className="mt-0.5 font-mono">{fmt(frame.ema_slow)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">RSI 14</div>
                <div className="mt-0.5 font-mono">{fmt(frame.rsi14, 1)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">ATR 14</div>
                <div className="mt-0.5 font-mono">{fmt(frame.atr14, 2)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Breakout</div>
                <div className="mt-0.5 uppercase">{frame.breakout}</div>
              </div>
              <div>
                <div className="text-muted-foreground">ATR %</div>
                <div className="mt-0.5 font-mono">{fmt(frame.atr_pct, 3)}%</div>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-2 border-t border-border/60 pt-3 text-[11px]">
              <div>
                <div className="text-muted-foreground">Swing high</div>
                <div className="font-mono">{fmt(frame.recent_swing_high)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Swing low</div>
                <div className="font-mono">{fmt(frame.recent_swing_low)}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
        <div className="card p-4 lg:col-span-7">
          <div className="mb-3 flex items-center gap-2">
            <Newspaper className="h-4 w-4 text-primary" />
            <h2 className="text-[14px] font-semibold">Macro & news context</h2>
            {macro?.event_risk && (
              <span className="ml-auto rounded-full bg-rose-500/10 px-2 py-1 text-[10px] font-semibold text-rose-500">
                EVENT RISK
              </span>
            )}
          </div>

          {macroLoading ? (
            <div className="py-8 text-center text-[12px] text-muted-foreground">Researching current macro context…</div>
          ) : macroError ? (
            <div className="rounded-xl bg-rose-500/8 p-3 text-[12px] text-rose-500">{macroError}</div>
          ) : (
            <>
              <p className="text-[13px] leading-6 text-foreground">{macro?.summary || 'No macro summary available.'}</p>
              <div className="mt-4 space-y-2">
                {(macro?.drivers || []).map((driver, index) => (
                  <div key={index} className="flex items-start gap-2 rounded-xl bg-accent/30 px-3 py-2.5 text-[12px] leading-5">
                    <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                    <span>{driver}</span>
                  </div>
                ))}
              </div>
              {!macro?.search_ok && (
                <div className="mt-3 text-[11px] text-muted-foreground">
                  External web research did not return usable material{macro?.search_error ? `: ${macro.search_error}` : '.'}
                </div>
              )}
            </>
          )}
        </div>

        <div className="space-y-3 lg:col-span-5">
          <div className="card p-4">
            <div className="mb-3 flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              <h2 className="text-[14px] font-semibold">Research levels</h2>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-xl bg-accent/30 p-3">
                <div className="text-[10px] text-muted-foreground">5m ATR</div>
                <div className="mt-1 font-mono text-[14px]">{fmt(snapshot?.atr_reference)}</div>
              </div>
              <div className="rounded-xl bg-accent/30 p-3">
                <div className="text-[10px] text-muted-foreground">Swing high</div>
                <div className="mt-1 font-mono text-[14px]">{fmt(snapshot?.swing_high_reference)}</div>
              </div>
              <div className="rounded-xl bg-accent/30 p-3">
                <div className="text-[10px] text-muted-foreground">Swing low</div>
                <div className="mt-1 font-mono text-[14px]">{fmt(snapshot?.swing_low_reference)}</div>
              </div>
            </div>
          </div>

          <div className="card p-4">
            <div className="mb-3 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-500" />
              <h2 className="text-[14px] font-semibold">Gates & warnings</h2>
            </div>
            {gateItems.length === 0 ? (
              <div className="text-[12px] text-muted-foreground">No research-data gates are active.</div>
            ) : (
              <div className="space-y-2">
                {gateItems.map((item) => (
                  <div key={item} className="rounded-xl bg-accent/30 px-3 py-2 text-[11px] text-muted-foreground">
                    {friendlyReason(item)}
                  </div>
                ))}
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => navigate('/assistant')}
            className="card flex w-full items-center justify-between p-4 text-left transition hover:border-primary/30 hover:bg-primary/5"
          >
            <div>
              <div className="flex items-center gap-2 text-[13px] font-semibold">
                <Sparkles className="h-4 w-4 text-primary" />
                Open AI research
              </div>
              <div className="mt-1 text-[11px] text-muted-foreground">
                Atria + Agent-Reach + Scrapling + XAU technical tool
              </div>
            </div>
            <span className="text-primary">→</span>
          </button>
        </div>
      </div>

      <div className="mt-4 text-[10px] leading-5 text-muted-foreground">
        {snapshot?.disclaimer || 'Research context only. A real spot execution feed is required before execution automation can be enabled.'}
      </div>
    </div>
  )
}
