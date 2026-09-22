import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, AlertTriangle, ArrowRight, Brain, Database, Lock, Newspaper, RefreshCw, Sparkles, Target } from 'lucide-react'
import { fetchAPI } from '@panwatch/api/client'
import { Button } from '@panwatch/base-ui/components/ui/button'
import XAUChart, { type XAUChartBar } from '@/components/XAUChart'

type Frame = { timeframe:string; close:number; ema_fast:number; ema_slow:number; rsi14:number; atr14:number; direction:string; recent_swing_high:number; recent_swing_low:number }
type Snapshot = { indicative_spot?:{price:number;bid:number|null;ask:number|null;spread_bps:number|null;source:string;is_stale:boolean}|null; micro?:{direction:string;return_10m_pct:number|null;return_30m_pct:number|null;source:string}|null; candidate:string; alignment:string; blocked:boolean; block_reasons:string[]; warnings:string[]; atr_reference:number|null; swing_high_reference:number|null; swing_low_reference:number|null; frames:Record<string,Frame>; disclaimer:string }
type Macro = { bias:number; bias_label:string; confidence:number; event_risk:boolean; summary:string; drivers:string[]; search_ok:boolean; synthesis_ok?:boolean; synthesis_error?:string|null; refresh_pending?:boolean }
type Hypothesis = { name:string; weight:number; direction:string }
type Scenario = { name:string; direction:string; weight:number; target:number|null; trigger:number|string|null; invalidation:number|null }
type Edge = { score:number; direction:string; strength:number; band:string; macro_freshness:number; components:Record<string,number> }
type Plan = { action:string; side:string|null; setup_confirmed:boolean; trigger_level:number|null; activation_conditions:string[]; invalidation_reference:number|null; reasons:string[] }
type Cognition = { version?:string; data_quality:{score:number;issues:string[]}; regime:{label:string;confidence:number}; hypotheses:Hypothesis[]; scenarios?:Scenario[]; directional_edge?:Edge; adversarial:{veto:boolean;counter_evidence:string[]}; confidence:{calibrated_confidence:number}; execution_plan:Plan; meta_controller:{decision:string} }
type Fusion = { state:string; macro_relation:string; research_ready:boolean; event_risk:boolean; reasons:string[]; cognition?:Cognition; execution_status:string }
type Terminal = { technical:Snapshot; macro:Macro; fusion:Fusion }
type Readiness = { profile:string; ai:{api_key_configured:boolean}; toolbox:{reachable:boolean;tool_count:number;scrapling_fetch_available:boolean} }
type ChartSeries = { instrument:string; timeframe:string; count:number; bars:XAUChartBar[]; source?:string|null; observed_at?:string|null }

const fmt=(v?:number|null,d=2)=>v==null||!Number.isFinite(v)?'--':v.toFixed(d)
const nice=(v?:string)=>String(v||'--').replace(/_/g,' ').replace(/^./,x=>x.toUpperCase())
const tone=(v?:string)=>v==='bullish'||v==='long_setup'?'text-emerald-500':v==='bearish'||v==='short_setup'?'text-rose-500':'text-muted-foreground'

export default function DashboardPage(){
 const nav=useNavigate(); const [snapshot,setSnapshot]=useState<Snapshot|null>(null); const [macro,setMacro]=useState<Macro|null>(null); const [fusion,setFusion]=useState<Fusion|null>(null); const [ready,setReady]=useState<Readiness|null>(null); const [chart,setChart]=useState<ChartSeries|null>(null); const [timeframe,setTimeframe]=useState<'1m'|'5m'|'15m'>('5m'); const [chartLoading,setChartLoading]=useState(true); const [loading,setLoading]=useState(true); const [error,setError]=useState('')
 const applyTerminal=useCallback((d:Terminal)=>{setSnapshot(d.technical);setMacro(d.macro);setFusion(d.fusion)},[])
 const load=useCallback(async(force=false)=>{setLoading(true);setError('');try{const d=await fetchAPI<Terminal>(`/xau/terminal${force?'?force=true':''}`,{timeoutMs:95000});applyTerminal(d)}catch(e){setError(e instanceof Error?e.message:'Research unavailable')}finally{setLoading(false)}},[applyTerminal])
 const refreshQuiet=useCallback(async()=>{try{const d=await fetchAPI<Terminal>('/xau/terminal',{timeoutMs:20000});applyTerminal(d)}catch{}},[applyTerminal])
 const loadChart=useCallback(async(tf:'1m'|'5m'|'15m',force=false,silent=false)=>{if(!silent)setChartLoading(true);try{const d=await fetchAPI<ChartSeries>(`/xau/chart?timeframe=${tf}&limit=160${force?'&force=true':''}`,{timeoutMs:45000});setChart(d)}catch{if(!silent)setChart(null)}finally{if(!silent)setChartLoading(false)}},[])
 useEffect(()=>{void load();fetch('/api/runtime-readiness').then(r=>r.json()).then(b=>setReady(b?.data||b)).catch(()=>{});const id=window.setInterval(()=>void refreshQuiet(),20000);return()=>window.clearInterval(id)},[load,refreshQuiet])
 useEffect(()=>{void loadChart(timeframe);const id=window.setInterval(()=>void loadChart(timeframe,false,true),20000);return()=>window.clearInterval(id)},[timeframe,loadChart])
 const frames=useMemo(()=>['1m','5m','15m'].map(k=>snapshot?.frames?.[k]).filter(Boolean) as Frame[],[snapshot])
 const cog=fusion?.cognition; const edge=cog?.directional_edge; const plan=cog?.execution_plan; const scenarios=cog?.scenarios||[]; const score=Math.round((cog?.confidence.calibrated_confidence||0)*100); const quality=Math.round((cog?.data_quality.score||0)*100)
 const price=snapshot?.indicative_spot?.price; const direction=edge?.direction||snapshot?.alignment||'mixed'
 const directionLabel=direction==='bullish'?'BULLISH LEAN':direction==='bearish'?'BEARISH LEAN':'NO CLEAR EDGE'
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

  <section className="mb-4 grid gap-3 xl:grid-cols-12">
   <div className="card overflow-hidden p-3 xl:col-span-8">
    <div className="mb-2 flex flex-wrap items-center justify-between gap-2 px-1">
     <div>
      <div className="text-sm font-semibold">Market structure</div>
      <div className="text-[10px] text-muted-foreground">{chart?.source||'research bars'} · {chart?.count||0} bars</div>
     </div>
     <div className="flex rounded-xl bg-accent/40 p-1">
      {(['1m','5m','15m'] as const).map(tf=><button key={tf} onClick={()=>setTimeframe(tf)} className={timeframe===tf?'rounded-lg bg-primary px-3 py-1.5 text-[11px] font-semibold text-primary-foreground shadow-sm':'rounded-lg px-3 py-1.5 text-[11px] font-semibold text-muted-foreground hover:text-foreground'}>{tf}</button>)}
     </div>
    </div>
    <div className="overflow-x-auto">
     <XAUChart bars={chart?.bars||[]} loading={chartLoading} swingHigh={snapshot?.swing_high_reference} swingLow={snapshot?.swing_low_reference} triggerLevel={plan?.trigger_level}/>
    </div>
   </div>
   <div className="space-y-3 xl:col-span-4">
    <div className="card p-4">
     <div className="text-[10px] uppercase tracking-[.15em] text-muted-foreground">Directional edge</div>
     <div className={'mt-1 text-xl font-bold '+tone(direction)}>{directionLabel}</div>
     <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
      <div className="rounded-xl bg-accent/30 p-3"><span className="text-muted-foreground">Edge score</span><div className="mt-1 font-mono font-semibold">{edge?Math.round(edge.score*100):'--'}</div></div>
      <div className="rounded-xl bg-accent/30 p-3"><span className="text-muted-foreground">Strength</span><div className="mt-1 font-semibold">{nice(edge?.band)}</div></div>
      <div className="rounded-xl bg-accent/30 p-3"><span className="text-muted-foreground">Trigger</span><div className="mt-1 font-mono font-semibold">{fmt(plan?.trigger_level)}</div></div>
      <div className="rounded-xl bg-accent/30 p-3"><span className="text-muted-foreground">Invalidation</span><div className="mt-1 font-mono font-semibold">{fmt(plan?.invalidation_reference)}</div></div>
     </div>
    </div>
    <div className="card p-4">
     <div className="flex items-center gap-2"><Target className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Activation conditions</h2></div>
     <div className="mt-3 space-y-2">{(plan?.activation_conditions||[]).map(x=><div key={x} className="flex items-center gap-2 text-[11px]"><span className="h-1.5 w-1.5 rounded-full bg-primary"/><span>{nice(x)}</span></div>)}{!(plan?.activation_conditions||[]).length&&<div className="text-[11px] text-muted-foreground">No directional trigger is active.</div>}</div>
    </div>
   </div>
  </section>

  <section className="mb-4 grid gap-3 lg:grid-cols-12">
   <div className="card p-5 lg:col-span-7">
    <div className="flex items-start justify-between gap-3">
     <div><div className="text-xs text-muted-foreground">PanWatch decision</div><div className={'mt-1 text-2xl font-bold '+tone(direction)}>{directionLabel}</div><div className="mt-1 text-sm text-muted-foreground">{statusText}</div><div className="mt-1 text-[10px] text-muted-foreground">{plan?.setup_confirmed?'Strict entry setup confirmed':'Directional bias only · entry trigger not yet confirmed'}</div></div>
     <div className="text-right"><div className="text-xs text-muted-foreground">Decision score</div><div className="mt-1 font-mono text-3xl font-bold">{score}<span className="text-sm text-muted-foreground"> / 100</span></div><div className="text-[10px] text-muted-foreground">not win probability</div></div>
    </div>
    <div className="mt-5 grid grid-cols-2 gap-2 md:grid-cols-4">
     {[['Regime',nice(cog?.regime.label)],['Data quality',quality+'%'],['Macro',macro?.synthesis_ok?nice(macro.bias_label):'Unavailable'],['Timing',statusText]].map(([a,b])=><div key={a} className="rounded-xl bg-accent/30 p-3"><div className="text-[10px] text-muted-foreground">{a}</div><div className="mt-1 text-xs font-semibold">{b}</div></div>)}
    </div>
   </div>
   <div className="card p-5 lg:col-span-5">
    <div className="flex items-center gap-2"><Brain className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Why now</h2></div>
    <div className="mt-4 space-y-2 text-xs">
     <div className="flex justify-between"><span className="text-muted-foreground">Directional edge</span><b className={tone(direction)}>{nice(direction)} {edge?Math.round(edge.strength*100)+'%':''}</b></div>
     <div className="flex justify-between"><span className="text-muted-foreground">Macro relation</span><b>{nice(fusion?.macro_relation)}</b></div>
     <div className="flex justify-between"><span className="text-muted-foreground">Adversarial check</span><b>{cog?.adversarial.veto?'VETO':'CLEAR'}</b></div>
     <div className="flex justify-between"><span className="text-muted-foreground">Execution</span><b className="text-amber-500">LOCKED</b></div>
    </div>
   </div>
  </section>

  <section className="mb-4 grid gap-3 md:grid-cols-3">
   {frames.map(f=><div key={f.timeframe} className="card p-4">
    <div className="flex items-center justify-between"><div><div className="text-[10px] uppercase tracking-[.15em] text-muted-foreground">{f.timeframe}</div><div className={`mt-1 font-bold uppercase ${tone(f.direction)}`}>{f.direction}</div></div><div className="font-mono text-xl font-semibold">{fmt(f.close)}</div></div>
    <div className="mt-4 grid grid-cols-4 gap-2 text-[10px]"><div><span className="text-muted-foreground">EMA9</span><div className="mt-1 font-mono">{fmt(f.ema_fast)}</div></div><div><span className="text-muted-foreground">EMA21</span><div className="mt-1 font-mono">{fmt(f.ema_slow)}</div></div><div><span className="text-muted-foreground">RSI</span><div className="mt-1 font-mono">{fmt(f.rsi14,1)}</div></div><div><span className="text-muted-foreground">ATR</span><div className="mt-1 font-mono">{fmt(f.atr14)}</div></div></div>
   </div>)}
  </section>

  <section className="mb-4 grid gap-3 lg:grid-cols-12">
   <div className="card p-5 lg:col-span-7">
    <div className="flex items-center justify-between"><div className="flex items-center gap-2"><Newspaper className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Macro & news context</h2></div><span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${macro?.synthesis_ok?'bg-emerald-500/10 text-emerald-500':'bg-amber-500/10 text-amber-500'}`}>{macro?.synthesis_ok?'SYNTHESIZED':'DEGRADED'}</span></div>
    <p className="mt-4 text-sm leading-6">{macro?.summary||'Macro research is refreshing.'}</p>
    <div className="mt-3 space-y-2">{(macro?.drivers||[]).map((d,i)=><div key={i} className="flex gap-2 rounded-xl bg-accent/25 px-3 py-2 text-xs"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary"/><span>{d}</span></div>)}</div>
    {!macro?.synthesis_ok&&macro?.search_ok&&<div className="mt-3 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-500">Web evidence is available; synthesis is temporarily degraded{macro?.synthesis_error?` (${macro.synthesis_error})`:''}.</div>}
   </div>
   <div className="space-y-3 lg:col-span-5">
    <div className="card p-5"><div className="flex items-center gap-2"><Target className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Key levels</h2></div><div className="mt-4 grid grid-cols-3 gap-2">{[['ATR 5m',snapshot?.atr_reference],['Swing high',snapshot?.swing_high_reference],['Swing low',snapshot?.swing_low_reference]].map(([a,b])=><div key={String(a)} className="rounded-xl bg-accent/30 p-3"><div className="text-[10px] text-muted-foreground">{String(a)}</div><div className="mt-1 font-mono text-sm">{fmt(b as number|null)}</div></div>)}</div></div>
    <button onClick={()=>nav('/assistant')} className="card flex w-full items-center justify-between p-5 text-left transition hover:border-primary/30 hover:bg-primary/5"><div><div className="flex items-center gap-2 text-sm font-semibold"><Sparkles className="h-4 w-4 text-primary"/>Deep AI Research</div><div className="mt-1 text-[11px] text-muted-foreground">Open the full research workspace</div></div><ArrowRight className="h-4 w-4 text-primary"/></button>
   </div>
  </section>

  <section className="mb-4 grid gap-3 lg:grid-cols-12">
   <div className="card p-5 lg:col-span-7"><div className="flex items-center gap-2"><Activity className="h-4 w-4 text-primary"/><h2 className="text-sm font-semibold">Competing hypotheses</h2></div><div className="mt-4 space-y-3">{(cog?.hypotheses||[]).map(h=><div key={h.name}><div className="mb-1 flex justify-between text-xs"><span>{nice(h.name)}</span><span className="font-mono">{Math.round(h.weight*100)}%</span></div><div className="h-1.5 overflow-hidden rounded-full bg-accent"><div className="h-full rounded-full bg-primary" style={{width:`${Math.max(2,Math.round(h.weight*100))}%`}}/></div></div>)}</div></div>
   <div className="card p-5 lg:col-span-5"><div className="flex items-center gap-2"><AlertTriangle className="h-4 w-4 text-amber-500"/><h2 className="text-sm font-semibold">Risk & gates</h2></div>{gates.length?<div className="mt-3 space-y-2">{gates.slice(0,5).map(x=><div key={x} className="rounded-xl bg-accent/30 px-3 py-2 text-xs text-muted-foreground">{nice(x)}</div>)}</div>:<div className="mt-4 text-xs text-muted-foreground">No research-data gates are active.</div>}<div className="mt-4 flex items-center gap-2 border-t border-border/60 pt-3 text-xs text-amber-500"><Lock className="h-3.5 w-3.5"/> Live execution remains locked until a tradable broker feed is connected.</div></div>
  </section>

  <section className="card p-4"><div className="flex flex-wrap items-center gap-x-8 gap-y-3 text-xs"><div className="flex items-center gap-2"><Database className="h-4 w-4 text-primary"/><b>Research stack</b></div><span>Atria <b className={ready?.ai.api_key_configured?'text-emerald-500':'text-rose-500'}>{ready?.ai.api_key_configured?'ONLINE':'OFFLINE'}</b></span><span>Agent-Reach <b className={ready?.toolbox.reachable?'text-emerald-500':'text-rose-500'}>{ready?.toolbox.reachable?`ONLINE · ${ready.toolbox.tool_count}`:'OFFLINE'}</b></span><span>Scrapling <b className={ready?.toolbox.scrapling_fetch_available?'text-emerald-500':'text-rose-500'}>{ready?.toolbox.scrapling_fetch_available?'ONLINE':'OFFLINE'}</b></span></div></section>
  <div className="mt-3 text-[10px] leading-5 text-muted-foreground">{snapshot?.disclaimer}</div>
 </div>
}
