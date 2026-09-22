import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, AlertTriangle, ArrowRight, Brain, Database, Lock, Newspaper, RefreshCw, Sparkles, Target } from 'lucide-react'
import { fetchAPI } from '@panwatch/api/client'
import { Button } from '@panwatch/base-ui/components/ui/button'
import XAUChart, { type XAUChartBar } from '@/components/XAUChart'

type Frame = { timeframe:string; close:number; ema_fast:number; ema_slow:number; rsi14:number; atr14:number; direction:string; recent_swing_high:number; recent_swing_low:number }
type BiasState = { direction:string; score:number; close?:number; ema?:Record<string,number|null>; available?:boolean }
type MarketContext = {
 bias:{monthly:BiasState;weekly:BiasState;daily:BiasState;h4:BiasState;h1:BiasState;composite_score:number;composite_direction:string;today_score:number;today_direction:string}
 volume_profile:{available:boolean;poc?:number;vah?:number;val?:number;location?:string;hvn?:number[];lvn?:number[];bins?:Array<{price:number;volume:number;share:number}>;source_type?:string;centralized_volume?:boolean}
 cash_flow:{available:boolean;direction?:string;score?:number;agreement?:string;spot_tick?:{available?:boolean;cmf20?:number;signed_tick_volume_imbalance?:number;score?:number;direction?:string};gc_futures?:{available?:boolean;cmf20?:number;signed_tick_volume_imbalance?:number;score?:number;direction?:string}}
 spot_tick_flow?:{available?:boolean;cmf20?:number;signed_tick_volume_imbalance?:number;score?:number;direction?:string}
 futures_flow?:{available?:boolean;cmf20?:number;signed_tick_volume_imbalance?:number;score?:number;direction?:string}
 liquidity:{available:boolean;levels?:Array<{name:string;price:number;side:string;distance:number}>;equal_highs?:number[];equal_lows?:number[]}
 smart_money:{available:boolean;bias?:string;score?:number;break_of_structure?:string;liquidity_sweep?:string;validated_liquidity_sweep?:string;validated_structure_direction?:string;displacement?:string;dealing_range?:{high:number;low:number;midpoint:number;zone:string};fair_value_gaps?:Array<{direction:string;low:number;high:number;time:string}>;library_primary?:{library?:string;status?:string;direction?:string;recent_liquidity_sweep?:string;latest_sweep_level?:number|null;fvg?:Record<string,unknown>;order_blocks?:Record<string,unknown>;zones?:Record<string,unknown>}}
 library_intelligence?:{version?:string;direction?:string;agreement?:number;independent_direction_votes?:number;status?:Record<string,string>;structure_scope_reference?:{status?:string;htf_direction?:string;session?:string;setup?:string}}
 cross_validation?:{smc_concordance?:string;library_direction?:string;library_agreement?:number;profile_parity?:{status?:string;poc_delta?:number;vah_delta?:number;val_delta?:number};technical_parity?:{status?:string;ema50_delta?:number|null}}
 volume_note?:string
}
type Snapshot = { indicative_spot?:{price:number;bid:number|null;ask:number|null;spread_bps:number|null;source:string;is_stale:boolean}|null; micro?:{direction:string;return_10m_pct:number|null;return_30m_pct:number|null;source:string}|null; candidate:string; alignment:string; blocked:boolean; block_reasons:string[]; warnings:string[]; atr_reference:number|null; swing_high_reference:number|null; swing_low_reference:number|null; frames:Record<string,Frame>; market_context?:MarketContext|null; market_context_error?:string|null; disclaimer:string }
type Macro = { bias:number; bias_label:string; confidence:number; event_risk:boolean; summary:string; drivers:string[]; search_ok:boolean; search_source?:string|null; synthesis_ok?:boolean; synthesis_error?:string|null; fallback_mode?:string|null; refresh_pending?:boolean }
type Hypothesis = { name:string; weight:number; direction:string }
type Scenario = { name:string; direction:string; weight:number; target:number|null; trigger:number|string|null; trigger_kind?:string; direction_basis?:string; weight_type?:string; invalidation:number|null }
type Edge = { score:number; direction:string; strength:number; band:string; macro_freshness:number; higher_timeframe_score?:number; higher_timeframe_direction?:string; higher_timeframe_conflict?:boolean; smart_money_score?:number; cash_flow_score?:number; components:Record<string,number> }
type ActivationCondition = { key:string; label:string; status:'satisfied'|'pending'|'failed'; current:number|string|null; threshold:number|string|null }
type ActivationState = { state:string; conditions:ActivationCondition[]; satisfied:number; pending:number; failed:number; total:number }
type Plan = { action:string; side:string|null; setup_confirmed:boolean; trigger_level:number|null; trigger_state?:string; activation?:ActivationState; activation_conditions:ActivationCondition[]; invalidation_reference:number|null; dominant_scenario?:string; dominant_scenario_direction?:string; dominant_scenario_weight?:number; scenario_conflict?:boolean; reasons:string[] }
type Cognition = { version?:string; data_quality:{score:number;issues:string[]}; regime:{label:string;confidence:number}; hypotheses:Hypothesis[]; scenarios?:Scenario[]; directional_edge?:Edge; adversarial:{veto:boolean;counter_evidence:string[]}; confidence:{calibrated_confidence:number}; execution_plan:Plan; meta_controller:{decision:string} }
type Fusion = { state:string; macro_relation:string; research_ready:boolean; event_risk:boolean; reasons:string[]; cognition?:Cognition; execution_status:string }
type Terminal = { technical:Snapshot; macro:Macro; fusion:Fusion }
type Readiness = { profile:string; ai:{api_key_configured:boolean}; toolbox:{reachable:boolean;tool_count:number;scrapling_fetch_available:boolean} }
type ChartSeries = { instrument:string; timeframe:string; count:number; bars:XAUChartBar[]; source?:string|null; observed_at?:string|null }

const fmt=(v?:number|null,d=2)=>v==null||!Number.isFinite(v)?'--':v.toFixed(d)
const nice=(v?:string)=>String(v||'--').replace(/_/g,' ').replace(/^./,x=>x.toUpperCase())
const tone=(v?:string)=>v==='bullish'||v==='long'||v==='long_setup'?'text-emerald-500':v==='bearish'||v==='short'||v==='short_setup'?'text-rose-500':'text-muted-foreground'
const conditionTone=(v?:string)=>v==='satisfied'?'text-emerald-500':v==='failed'?'text-rose-500':'text-amber-500'
const conditionDot=(v?:string)=>v==='satisfied'?'bg-emerald-500':v==='failed'?'bg-rose-500':'bg-amber-500'

export default function DashboardPage(){
 const nav=useNavigate(); const [snapshot,setSnapshot]=useState<Snapshot|null>(null); const [macro,setMacro]=useState<Macro|null>(null); const [fusion,setFusion]=useState<Fusion|null>(null); const [ready,setReady]=useState<Readiness|null>(null); const [chart,setChart]=useState<ChartSeries|null>(null); const [timeframe,setTimeframe]=useState<'1m'|'5m'|'15m'|'30m'|'1h'|'4h'|'1d'|'1w'|'1mo'>('15m'); const [chartLoading,setChartLoading]=useState(true); const [loading,setLoading]=useState(true); const [error,setError]=useState('')
 const applyTerminal=useCallback((d:Terminal)=>{setSnapshot(d.technical);setMacro(d.macro);setFusion(d.fusion)},[])
 const load=useCallback(async(force=false)=>{setLoading(true);setError('');try{const d=await fetchAPI<Terminal>(`/xau/terminal${force?'?force=true':''}`,{timeoutMs:95000});applyTerminal(d)}catch(e){setError(e instanceof Error?e.message:'Research unavailable')}finally{setLoading(false)}},[applyTerminal])
 const refreshQuiet=useCallback(async()=>{try{const d=await fetchAPI<Terminal>('/xau/terminal',{timeoutMs:20000});applyTerminal(d)}catch{}},[applyTerminal])
 const loadChart=useCallback(async(tf:'1m'|'5m'|'15m'|'30m'|'1h'|'4h'|'1d'|'1w'|'1mo',force=false,silent=false)=>{if(!silent)setChartLoading(true);try{const d=await fetchAPI<ChartSeries>(`/xau/chart?timeframe=${tf}&limit=160${force?'&force=true':''}`,{timeoutMs:45000});setChart(d)}catch{if(!silent)setChart(null)}finally{if(!silent)setChartLoading(false)}},[])
 useEffect(()=>{void load();fetch('/api/runtime-readiness').then(r=>r.json()).then(b=>setReady(b?.data||b)).catch(()=>{});const id=window.setInterval(()=>void refreshQuiet(),20000);return()=>window.clearInterval(id)},[load,refreshQuiet])
 useEffect(()=>{void loadChart(timeframe);const id=window.setInterval(()=>void loadChart(timeframe,false,true),20000);return()=>window.clearInterval(id)},[timeframe,loadChart])
 const cog=fusion?.cognition; const edge=cog?.directional_edge; const plan=cog?.execution_plan; const scenarios=cog?.scenarios||[]; const score=Math.round((cog?.confidence.calibrated_confidence||0)*100); const quality=Math.round((cog?.data_quality.score||0)*100)
 const context=snapshot?.market_context
 const price=snapshot?.indicative_spot?.price; const direction=edge?.direction||snapshot?.alignment||'mixed'
 const directionLabel=plan?.scenario_conflict?'CONFLICTED — WAIT':direction==='bullish'?(edge?.strength!=null&&edge.strength<0.24?'WEAK BULLISH BIAS':'BULLISH LEAN'):direction==='bearish'?(edge?.strength!=null&&edge.strength<0.24?'WEAK BEARISH BIAS':'BEARISH LEAN'):'NO CLEAR EDGE'
 const statusText=plan?.action?nice(plan.action):nice(fusion?.state)
 const gates=[...(snapshot?.block_reasons||[]),...(snapshot?.warnings||[]),...(cog?.adversarial.counter_evidence||[])]

 return <div className="page-container pb-10">
  <section className="mb-4 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
   <div>
    <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[.16em] text-muted-foreground"><span className="h-2 w-2 rounded-full bg-emerald-500"/> Live research terminal</div>
    <div className="mt-2 flex flex-wrap items-end gap-x-5 gap-y-1"><h1 className="text-3xl font-bold tracking-tight">XAU/USD</h1><div className="font-mono text-3xl font-bold">{fmt(price)}</div><div className={`pb-1 text-sm font-semibold uppercase ${tone(direction)}`}>{direction}</div></div>
    <p className="mt-1 text-xs text-muted-foreground">Gold intelligence · technical structure · macro context · adversarial reasoning</p>
   </div>
   <Button className="self-start lg:self-auto" onClick={()=>void load(true)} disabled={loading}><RefreshCw className={`mr-2 h-4 w-4 ${loading?'animate-spin':''}`}/>Refresh intelligence</Button>
  </section>

  {error&&<div className="mb-4 rounded-2xl border border-rose-500/20 bg-rose-500/10 p-3 text-sm text-rose-500">{error}</div>}

  <section className="mb-4 grid items-start gap-3 lg:grid-cols-12">
   <div className="card self-start overflow-hidden p-3 lg:col-span-8">
    <div className="mb-2 flex flex-wrap items-center justify-between gap-2 px-1">
     <div>
      <div className="text-sm font-semibold">Market structure</div>
      <div className="text-[10px] text-muted-foreground">{chart?.source||'research bars'} · {chart?.count||0} bars</div>
     </div>
     <div className="flex rounded-xl bg-accent/40 p-1">
      {(['1m','5m','15m','30m','1h','4h','1d','1w','1mo'] as const).map(tf=><button key={tf} onClick={()=>setTimeframe(tf)} className={timeframe===tf?'rounded-lg bg-primary px-3 py-1.5 text-[11px] font-semibold text-primary-foreground shadow-sm':'rounded-lg px-3 py-1.5 text-[11px] font-semibold text-muted-foreground hover:text-foreground'}>{tf}</button>)}
     </div>
    </div>
    <div className="overflow-x-auto">
     <XAUChart bars={chart?.bars||[]} loading={chartLoading} swingHigh={snapshot?.swing_high_reference} swingLow={snapshot?.swing_low_reference} triggerLevel={plan?.trigger_level} volumeProfile={context?.volume_profile} liquidityLevels={context?.liquidity.levels||[]}/>
    </div>
   </div>

   <aside className="space-y-3 lg:col-span-4">
    <div className="card p-4">
     <div className="flex items-start justify-between gap-3">
      <div><div className="text-[10px] uppercase tracking-[.15em] text-muted-foreground">PanWatch decision</div><div className={'mt-1 text-xl font-bold '+(plan?.scenario_conflict?'text-amber-500':tone(direction))}>{directionLabel}</div><div className="mt-1 text-[11px] font-medium text-muted-foreground">{statusText}</div></div>
      <div className="text-right"><div className="font-mono text-2xl font-bold">{score}<span className="text-[11px] text-muted-foreground">/100</span></div><div className="text-[9px] text-muted-foreground">decision score</div></div>
     </div>
     <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
      <div className="rounded-xl bg-accent/30 p-3"><span className="text-muted-foreground">Edge</span><div className={'mt-1 font-semibold '+tone(direction)}>{edge?Math.round(edge.score*100):'--'} · {nice(edge?.band)}</div></div>
      <div className="rounded-xl bg-accent/30 p-3"><span className="text-muted-foreground">Regime</span><div className="mt-1 font-semibold">{nice(cog?.regime.label)}</div></div>
      <div className="rounded-xl bg-accent/30 p-3"><span className="text-muted-foreground">Trigger</span><div className="mt-1 font-mono font-semibold">{fmt(plan?.trigger_level)}</div><div className="mt-0.5 text-[9px] text-muted-foreground">{nice(plan?.trigger_state)}</div></div>
      <div className="rounded-xl bg-accent/30 p-3"><span className="text-muted-foreground">Invalidation</span><div className="mt-1 font-mono font-semibold">{fmt(plan?.invalidation_reference)}</div></div>
     </div>
     <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
      <div className="flex justify-between rounded-lg bg-accent/20 px-2.5 py-2"><span className="text-muted-foreground">Data quality</span><b>{quality}%</b></div>
      <div className="flex justify-between rounded-lg bg-accent/20 px-2.5 py-2"><span className="text-muted-foreground">Macro</span><b>{macro?.synthesis_ok?nice(macro.bias_label):'Degraded'}</b></div>
     </div>
     {plan?.scenario_conflict&&<div className="mt-3 rounded-xl border border-amber-500/20 bg-amber-500/5 p-2.5 text-[10px] leading-5 text-amber-500">Dominant scenario conflicts with the directional edge. PanWatch is waiting for confirmation instead of forcing an entry.</div>}
    </div>

    <div className="card p-4">
     <div className="flex items-center justify-between gap-2"><div className="flex items-center gap-2"><Target className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Activation state</h2></div><span className="text-[10px] text-muted-foreground">{plan?.activation?.satisfied||0}/{plan?.activation?.total||0} satisfied</span></div>
     <div className="mt-3 space-y-2">{(plan?.activation_conditions||[]).map(x=><div key={x.key} className="flex items-center justify-between gap-3 rounded-lg bg-accent/20 px-3 py-2 text-[11px]"><div className="flex min-w-0 items-center gap-2"><span className={'h-1.5 w-1.5 shrink-0 rounded-full '+conditionDot(x.status)}/><span className="truncate">{x.label}</span></div><span className={'shrink-0 text-[10px] font-semibold '+conditionTone(x.status)}>{x.status.toUpperCase()}</span></div>)}{!(plan?.activation_conditions||[]).length&&<div className="text-[11px] text-muted-foreground">No directional trigger is active.</div>}</div>
    </div>
   </aside>
  </section>

  <section className="mb-4 grid gap-3 lg:grid-cols-12">
   <div className="card p-4 lg:col-span-7">
    <div className="flex items-center justify-between gap-3"><div><div className="text-sm font-semibold">Top-down bias</div><div className="mt-1 text-[10px] text-muted-foreground">Monthly → weekly → daily → 4H → 1H · EMA 9/21/50/200/1000</div></div><div className={'text-sm font-bold '+tone(context?.bias.today_direction)}>{nice(context?.bias.today_direction)} {context?Math.round(context.bias.today_score*100):'--'}</div></div>
    <div className="mt-4 grid grid-cols-5 gap-2">
     {[
       ['1M',context?.bias.monthly],
       ['1W',context?.bias.weekly],
       ['1D',context?.bias.daily],
       ['4H',context?.bias.h4],
       ['1H',context?.bias.h1],
     ].map(([label,state])=>{const s=state as BiasState|undefined;return <div key={String(label)} className="rounded-xl bg-accent/25 p-3"><div className="text-[10px] text-muted-foreground">{String(label)}</div><div className={'mt-1 text-xs font-semibold '+tone(s?.direction)}>{nice(s?.direction)}</div><div className="mt-1 font-mono text-[10px] text-muted-foreground">{s?Math.round(s.score*100):'--'}</div></div>})}
    </div>
    <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
     {[
       ['Monthly',context?.bias.monthly],
       ['Weekly',context?.bias.weekly],
       ['Daily',context?.bias.daily],
       ['4H',context?.bias.h4],
       ['1H',context?.bias.h1],
     ].map(([label,state])=>{const s=state as BiasState|undefined;return <div key={String(label)} className="rounded-xl border border-border/60 p-3"><div className="text-[10px] font-semibold">{String(label)} EMA ladder</div><div className="mt-2 grid grid-cols-3 gap-x-2 gap-y-1 text-[9px] font-mono text-muted-foreground"><span>50 {fmt(s?.ema?.['50'])}</span><span>200 {fmt(s?.ema?.['200'])}</span><span>1000 {fmt(s?.ema?.['1000'])}</span></div></div>})}
    </div>
   </div>

   <div className="card p-4 lg:col-span-5">
    <div className="flex items-center justify-between"><div><div className="text-sm font-semibold">Liquidity & smart money</div><div className="mt-1 text-[10px] text-muted-foreground">Structure + participation + volume-at-price proxy</div></div><div className={'text-xs font-bold '+tone(context?.smart_money.bias)}>{nice(context?.smart_money.bias)}</div></div>
    <div className="mt-4 grid grid-cols-3 gap-2 text-[10px]">
     <div className="rounded-xl bg-accent/25 p-3"><span className="text-muted-foreground">POC</span><div className="mt-1 font-mono text-xs">{fmt(context?.volume_profile.poc)}</div></div>
     <div className="rounded-xl bg-accent/25 p-3"><span className="text-muted-foreground">VAH</span><div className="mt-1 font-mono text-xs">{fmt(context?.volume_profile.vah)}</div></div>
     <div className="rounded-xl bg-accent/25 p-3"><span className="text-muted-foreground">VAL</span><div className="mt-1 font-mono text-xs">{fmt(context?.volume_profile.val)}</div></div>
    </div>
    <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
     <div className="rounded-xl border border-border/60 p-3"><span className="text-muted-foreground">Cash flow</span><div className={'mt-1 text-xs font-semibold '+tone(context?.cash_flow.direction==='inflow'?'bullish':context?.cash_flow.direction==='outflow'?'bearish':'neutral')}>{nice(context?.cash_flow.direction)} · {fmt(context?.cash_flow.score,2)}</div><div className="mt-1 font-mono text-[9px] text-muted-foreground">Spot CMF {fmt(context?.spot_tick_flow?.cmf20,3)} · GC CMF {fmt(context?.futures_flow?.cmf20,3)}</div><div className="mt-1 text-[9px] text-muted-foreground">{nice(context?.cash_flow.agreement)}</div></div>
     <div className="rounded-xl border border-border/60 p-3"><span className="text-muted-foreground">Structure</span><div className="mt-1 text-xs font-semibold">{context?.smart_money.break_of_structure&&context.smart_money.break_of_structure!=='none'?nice(context.smart_money.break_of_structure):'Validated '+nice(context?.smart_money.validated_structure_direction)}</div><div className="mt-1 text-[9px] text-muted-foreground">Sweep {nice(context?.smart_money.validated_liquidity_sweep||context?.smart_money.liquidity_sweep)} · {nice(context?.smart_money.dealing_range?.zone)}</div></div>
    </div>
    <div className="mt-3 flex flex-wrap gap-2">{(context?.liquidity.levels||[]).slice(0,6).map(level=><span key={level.name} className="rounded-full bg-accent/35 px-2.5 py-1 text-[9px]"><b>{level.name}</b> <span className="font-mono">{fmt(level.price)}</span></span>)}</div>
    <div className="mt-3 rounded-xl border border-border/60 p-3">
     <div className="flex items-center justify-between gap-3"><span className="text-[10px] text-muted-foreground">SMC cross-validation</span><b className={'text-[10px] '+tone(context?.library_intelligence?.direction)}>{nice(context?.library_intelligence?.direction)} · {context?.library_intelligence?.independent_direction_votes!=null?context.library_intelligence.independent_direction_votes+'/3 validators':'--'}</b></div>
     <div className="mt-2 flex flex-wrap gap-1.5">{Object.entries(context?.library_intelligence?.status||{}).map(([name,status])=><span key={name} className={'rounded-full px-2 py-1 text-[8px] '+(status==='ok'?'bg-emerald-500/10 text-emerald-500':'bg-amber-500/10 text-amber-500')}>{name} {String(status).toUpperCase()}</span>)}</div>
     <div className="mt-2 grid grid-cols-2 gap-2 text-[9px] text-muted-foreground"><span>SMC: {nice(context?.cross_validation?.smc_concordance)}</span><span>Structure-scope: {nice(context?.library_intelligence?.structure_scope_reference?.setup)}</span><span>TA EMA50 Δ {fmt(context?.cross_validation?.technical_parity?.ema50_delta,4)}</span><span>VP {nice((context?.cross_validation as any)?.profile_parity?.concordance)}</span></div>
    </div>
    <div className="mt-3 text-[9px] leading-4 text-muted-foreground">{context?.volume_note||'Higher-timeframe context is loading.'}</div>
   </div>
  </section>

  <section className="mb-4 grid gap-3 lg:grid-cols-12">
   <div className="card p-5 lg:col-span-7">
    <div className="flex items-center justify-between"><div className="flex items-center gap-2"><Newspaper className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Macro & news context</h2></div><span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${macro?.synthesis_ok?'bg-emerald-500/10 text-emerald-500':'bg-amber-500/10 text-amber-500'}`}>{macro?.synthesis_ok?'SYNTHESIZED':macro?.search_ok?'EVIDENCE ONLY':'DEGRADED'}</span></div>
    <p className="mt-4 text-sm leading-6">{macro?.summary||'Macro research is refreshing.'}</p>
    <div className="mt-3 space-y-2">{(macro?.drivers||[]).map((d,i)=><div key={i} className="flex gap-2 rounded-xl bg-accent/25 px-3 py-2 text-xs"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary"/><span>{d}</span></div>)}</div>
    {!macro?.synthesis_ok&&macro?.search_ok&&<div className="mt-3 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-500">Fresh macro evidence is available; synthesis is temporarily degraded{macro?.synthesis_error?` (${macro.synthesis_error})`:''}.</div>}
   </div>
   <div className="space-y-3 lg:col-span-5">
    <div className="card p-5"><div className="flex items-center gap-2"><Target className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Key levels</h2></div><div className="mt-4 grid grid-cols-3 gap-2">{[['ATR 5m',snapshot?.atr_reference],['Swing high',snapshot?.swing_high_reference],['Swing low',snapshot?.swing_low_reference]].map(([a,b])=><div key={String(a)} className="rounded-xl bg-accent/30 p-3"><div className="text-[10px] text-muted-foreground">{String(a)}</div><div className="mt-1 font-mono text-sm">{fmt(b as number|null)}</div></div>)}</div></div>
    <button onClick={()=>nav('/assistant')} className="card flex w-full items-center justify-between p-5 text-left transition hover:border-primary/30 hover:bg-primary/5"><div><div className="flex items-center gap-2 text-sm font-semibold"><Sparkles className="h-4 w-4 text-primary"/>Deep AI Research</div><div className="mt-1 text-[11px] text-muted-foreground">Open the full research workspace</div></div><ArrowRight className="h-4 w-4 text-primary"/></button>
   </div>
  </section>

  <section className="mb-4 grid gap-3 lg:grid-cols-12">
   <div className="card p-5 lg:col-span-5">
    <div className="flex items-center justify-between gap-2"><div className="flex items-center gap-2"><Brain className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Scenario paths</h2></div><span className="text-[9px] text-muted-foreground">relative weights · not probabilities</span></div>
    <div className="mt-4 space-y-2">{scenarios.map(s=><div key={s.name} className="rounded-xl border border-border/60 p-3"><div className="flex items-center justify-between"><div><div className="text-xs font-semibold">{nice(s.name)}</div><div className={'mt-1 text-[10px] '+tone(s.direction)}>{s.direction==='neutral'?'Direction unresolved':nice(s.direction)}</div><div className="mt-0.5 text-[9px] text-muted-foreground">{nice(s.direction_basis)}</div></div><div className="font-mono text-sm font-semibold">{Math.round(s.weight*100)}%</div></div><div className="mt-2 grid grid-cols-3 gap-2 text-[10px]"><div><span className="text-muted-foreground">Target</span><div className="font-mono">{fmt(s.target)}</div></div><div><span className="text-muted-foreground">Trigger</span><div className="font-mono">{typeof s.trigger==='number'?fmt(s.trigger):typeof s.trigger==='string'?nice(s.trigger):'--'}</div></div><div><span className="text-muted-foreground">Invalid.</span><div className="font-mono">{fmt(s.invalidation)}</div></div></div></div>)}</div>
   </div>
   <div className="card p-5 lg:col-span-4"><div className="flex items-center justify-between gap-2"><div className="flex items-center gap-2"><Activity className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Competing hypotheses</h2></div><span className="text-[9px] text-muted-foreground">relative weights</span></div><div className="mt-4 space-y-3">{(cog?.hypotheses||[]).map(h=><div key={h.name}><div className="mb-1 flex justify-between text-xs"><span>{nice(h.name)}</span><span className="font-mono">{Math.round(h.weight*100)}%</span></div><div className="h-1.5 overflow-hidden rounded-full bg-accent"><div className="h-full rounded-full bg-primary" style={{width:(Math.max(2,Math.round(h.weight*100)))+'%'}}/></div></div>)}</div></div>
   <div className="card p-5 lg:col-span-3"><div className="flex items-center gap-2"><AlertTriangle className="h-4 w-4 text-amber-500"/><h2 className="text-sm font-semibold">Risk & gates</h2></div>{gates.length?<div className="mt-3 space-y-2">{Array.from(new Set(gates)).slice(0,5).map(x=><div key={x} className="rounded-xl bg-accent/30 px-3 py-2 text-[11px] text-muted-foreground">{nice(x)}</div>)}</div>:<div className="mt-4 text-xs text-muted-foreground">No research-data gates are active.</div>}<div className="mt-4 flex items-center gap-2 border-t border-border/60 pt-3 text-[11px] text-amber-500"><Lock className="h-3.5 w-3.5"/> Live execution remains locked until a tradable broker feed is connected.</div></div>
  </section>

  <section className="card p-4"><div className="flex flex-wrap items-center gap-x-8 gap-y-3 text-xs"><div className="flex items-center gap-2"><Database className="h-4 w-4 text-primary"/><b>Research stack</b></div><span>Atria <b className={ready?.ai.api_key_configured?'text-emerald-500':'text-rose-500'}>{ready?.ai.api_key_configured?'ONLINE':'OFFLINE'}</b></span><span>Agent-Reach <b className={ready?.toolbox.reachable?'text-emerald-500':'text-rose-500'}>{ready?.toolbox.reachable?`ONLINE · ${ready.toolbox.tool_count}`:'OFFLINE'}</b></span><span>Scrapling <b className={ready?.toolbox.scrapling_fetch_available?'text-emerald-500':'text-rose-500'}>{ready?.toolbox.scrapling_fetch_available?'ONLINE':'OFFLINE'}</b></span></div></section>
  <div className="mt-3 text-[10px] leading-5 text-muted-foreground">{snapshot?.disclaimer}</div>
 </div>
}
