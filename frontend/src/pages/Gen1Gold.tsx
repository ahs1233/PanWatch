import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, CheckCircle2, Crosshair, RefreshCw, ShieldAlert, Waves } from 'lucide-react'
import { fetchAPI } from '@panwatch/api/client'
import { Button } from '@panwatch/base-ui/components/ui/button'

type AnyMap = Record<string, any>
type Gen1GoldPayload = {
  contract:string; pipeline_status:string; decision:string
  stages:{ ahmed_toolbox?:AnyMap; panwatch?:AnyMap; gen1?:AnyMap }
  missing_layers:string[]; stage_errors:Record<string,string>
  macro:AnyMap; technical:AnyMap; fusion:AnyMap; forward_range_map?:AnyMap|null
}
const fmt=(value:any,digits=2)=>{const n=Number(value);return Number.isFinite(n)?n.toFixed(digits):'—'}

function Health({ok,label,detail}:{ok:boolean;label:string;detail?:string}) {
  return <div className="flex items-center justify-between gap-3 rounded-xl border border-border/60 bg-background/35 px-3 py-2.5">
    <div className="flex min-w-0 items-center gap-2">
      {ok?<CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500"/>:<AlertTriangle className="h-4 w-4 shrink-0 text-amber-500"/>}
      <span className="text-xs font-medium text-foreground">{label}</span>
    </div>
    <span className="truncate text-[10px] text-muted-foreground">{detail||(ok?'Ready':'Degraded')}</span>
  </div>
}
function Metric({label,value,hint}:{label:string;value:React.ReactNode;hint?:string}) {
  return <div className="rounded-2xl border border-border/60 bg-background/30 p-3">
    <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
    <div className="mt-1 text-lg font-semibold text-foreground">{value}</div>
    {hint&&<div className="mt-1 text-[10px] text-muted-foreground">{hint}</div>}
  </div>
}

export default function Gen1GoldPage(){
  const [data,setData]=useState<Gen1GoldPayload|null>(null)
  const [loading,setLoading]=useState(true)
  const [error,setError]=useState('')
  const load=useCallback(async()=>{
    setLoading(true);setError('')
    try{setData(await fetchAPI<Gen1GoldPayload>('/xau/gen1-gold',{timeoutMs:120000}))}
    catch(err:any){setError(err?.message||'GEN1 GOLD unavailable')}
    finally{setLoading(false)}
  },[])
  useEffect(()=>{void load()},[load])

  const technical=data?.technical||{}
  const fusion=data?.fusion||{}
  const xaut=technical?.xaut_order_flow||{}
  const profile=xaut?.volume_profile||{}
  const flow5=xaut?.flow?.['5m']||{}
  const book10=xaut?.raw_book?.pm10||{}
  const market=technical?.market_context||{}
  const bias=market?.bias||{}
  const frames=useMemo(()=>[
    ['Monthly',bias.monthly],['Weekly',bias.weekly],['Daily',bias.daily],['H4',bias.h4],['H1',bias.h1],
  ],[bias])
  const stages=data?.stages||{}
  const health=[
    [stages.ahmed_toolbox?.status==='ready','Ahmed Toolbox',stages.ahmed_toolbox?.search_source],
    [Boolean(stages.panwatch?.market_context_ready),'HTF Market Context',stages.panwatch?.status],
    [Boolean(stages.panwatch?.xaut_ready),'XAUT Order Flow',xaut?.transport],
    [Boolean(stages.panwatch?.footprint_ready),'Footprint',xaut?.footprint?.method],
    [Boolean(stages.panwatch?.volume_profile_ready),'Volume Profile',profile?.volume_kind],
    [Boolean(stages.panwatch?.raw_book_ready),'Raw Order Book','Bitfinex R0'],
  ] as Array<[boolean,string,string|undefined]>
  const decisionClass=data?.decision==='LONG'?'text-emerald-500':data?.decision==='SHORT'?'text-rose-500':'text-amber-500'

  return <div className="mx-auto w-full max-w-[1500px] space-y-4 pb-8">
    <div className="card p-4 md:p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-primary"><Crosshair className="h-4 w-4"/> GEN1 GOLD</div>
          <h1 className="mt-1 text-2xl font-bold text-foreground">Gold Intelligence Room</h1>
          <p className="mt-1 text-xs text-muted-foreground">Same core as “Gen1 trade gold”: Ahmed Toolbox → PanWatch → Gen1.</p>
        </div>
        <Button onClick={()=>void load()} disabled={loading} variant="secondary"><RefreshCw className={`mr-2 h-4 w-4 ${loading?'animate-spin':''}`}/> Refresh</Button>
      </div>
    </div>
    {error&&<div className="card border-rose-500/30 p-4 text-sm text-rose-500">{error}</div>}
    {!data&&loading&&<div className="card p-8 text-center text-sm text-muted-foreground">Running Ahmed Toolbox → PanWatch → Gen1…</div>}
    {data&&<>
      <div className="grid gap-3 md:grid-cols-4">
        <Metric label="GEN1 Decision" value={<span className={decisionClass}>{data.decision||'WAIT'}</span>} hint={fusion?.state||'—'}/>
        <Metric label="Confidence" value={fusion?.cognitive_confidence==null?'—':`${Math.round(Number(fusion.cognitive_confidence)*100)}%`} hint={fusion?.meta_decision||'—'}/>
        <Metric label="Regime" value={fusion?.regime||'—'} hint={fusion?.macro_relation||'—'}/>
        <Metric label="Pipeline" value={data.pipeline_status?.toUpperCase()||'—'} hint={data.contract}/>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.15fr_1fr]">
        <section className="card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold"><Waves className="h-4 w-4 text-primary"/> Market State</div>
          <div className="grid gap-2 sm:grid-cols-5">
            {frames.map(([name,state]:any)=><div key={name} className="rounded-xl border border-border/60 p-3">
              <div className="text-[10px] text-muted-foreground">{name}</div>
              <div className="mt-1 text-sm font-semibold capitalize">{state?.direction||state?.bias||'unavailable'}</div>
              <div className="mt-1 text-[10px] text-muted-foreground">score {fmt(state?.score,3)}</div>
            </div>)}
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            <Metric label="XAUT CVD" value={fmt(xaut?.cvd?.value,4)} hint={`5m Δ ${fmt(xaut?.cvd?.change_5m,4)}`}/>
            <Metric label="5m Delta" value={fmt(flow5?.delta,4)} hint={`ratio ${fmt(flow5?.delta_ratio,3)}`}/>
            <Metric label="Book ±10" value={fmt(book10?.imbalance,3)} hint={`bid ${fmt(book10?.bid_quantity,3)} / ask ${fmt(book10?.ask_quantity,3)}`}/>
          </div>
        </section>
        <section className="card p-4">
          <div className="mb-3 text-sm font-semibold">Volume Profile — executed XAUT volume</div>
          <div className="grid grid-cols-3 gap-2">
            <Metric label="VAH" value={fmt(profile?.vah)}/><Metric label="POC" value={fmt(profile?.poc)}/><Metric label="VAL" value={fmt(profile?.val)}/>
          </div>
          <div className="mt-3 rounded-xl border border-border/60 p-3 text-xs text-muted-foreground">
            Location: <span className="font-medium text-foreground">{profile?.location||'—'}</span> · volume kind: {profile?.volume_kind||'—'} · global XAUUSD profile: <span className="font-medium">No</span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-xl bg-accent/30 p-3">HVN: {(profile?.high_volume_nodes||[]).slice(0,4).map((x:any)=>fmt(x?.price??x)).join(' · ')||'—'}</div>
            <div className="rounded-xl bg-accent/30 p-3">LVN: {(profile?.low_volume_nodes||[]).slice(0,4).map((x:any)=>fmt(x?.price??x)).join(' · ')||'—'}</div>
          </div>
        </section>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <section className="card p-4">
          <div className="mb-3 text-sm font-semibold">±10 / ±20 / ±30 scenarios</div>
          <div className="grid gap-2 sm:grid-cols-3">
            {[10,20,30].map(distance=>{const row=data.forward_range_map?.[`pm${distance}`]||{};return <div key={distance} className="rounded-2xl border border-border/60 p-3">
              <div className="text-[10px] text-muted-foreground">±{distance}</div>
              <div className="mt-2 text-xs text-emerald-500">UP {fmt(row.xau_up)}</div>
              <div className="mt-1 text-xs text-rose-500">DOWN {fmt(row.xau_down)}</div>
            </div>})}
          </div>
          <div className="mt-3 rounded-xl border border-border/60 p-3 text-xs text-muted-foreground">Gen1 reasons: {(fusion?.reasons||[]).join(' · ')||'—'}</div>
        </section>
        <section className="card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold"><ShieldAlert className="h-4 w-4 text-primary"/> Pipeline Health</div>
          <div className="grid gap-2 sm:grid-cols-2">{health.map(([ok,label,detail])=><Health key={label} ok={ok} label={label} detail={detail}/>)}</div>
          {!!data.missing_layers?.length&&<div className="mt-3 rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-amber-600">Missing: {data.missing_layers.join(' · ')}</div>}
          {!!Object.keys(data.stage_errors||{}).length&&<div className="mt-2 rounded-xl border border-rose-500/30 bg-rose-500/5 p-3 text-xs text-rose-500">Errors: {Object.entries(data.stage_errors).map(([k,v])=>`${k}=${v}`).join(' · ')}</div>}
        </section>
      </div>
      <div className="flex items-center gap-2 px-1 text-[10px] text-muted-foreground"><ShieldAlert className="h-3.5 w-3.5"/> Research-only. XAUT is a centralized gold proxy; live XAUUSD execution remains disabled.</div>
    </>}
  </div>
}
