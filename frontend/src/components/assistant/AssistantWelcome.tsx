import { ArrowUpRight, Briefcase, Search, Sparkles, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { fetchAPI } from '@panwatch/api/client'

interface AssistantWelcomeProps {
  onSubmit: (question: string) => void
  disabled?: boolean
}

interface RuntimeReadiness {
  status: 'ready' | 'partial'
  ai: {
    model: string
    api_key_configured: boolean
  }
  toolbox: {
    configured: boolean
    reachable: boolean
    tool_count: number
    error: string | null
  }
}

const QUICK_QUESTIONS = [
  { label: 'Analyze an asset', question: 'Analyze XAUUSD using current market context, recent news and technical structure.', icon: Search },
  { label: 'Diagnose my portfolio', question: 'Diagnose my portfolio risk and identify the most important things I should monitor.', icon: Briefcase },
  { label: 'Find today\'s opportunities', question: 'Use today\'s market conditions to find opportunities worth researching.', icon: Sparkles },
]

/** First-run surface for the full-page assistant before a conversation exists. */
export function AssistantWelcome({ onSubmit, disabled = false }: AssistantWelcomeProps) {
  const [question, setQuestion] = useState('')
  const [readiness, setReadiness] = useState<RuntimeReadiness | null>(null)

  useEffect(() => {
    let active = true
    fetchAPI<RuntimeReadiness>('/runtime-readiness')
      .then((result) => {
        if (active) setReadiness(result)
      })
      .catch(() => {
        if (active) setReadiness(null)
      })
    return () => {
      active = false
    }
  }, [])

  const readinessMessage = useMemo(() => {
    if (!readiness) return null
    if (!readiness.ai.api_key_configured) {
      return `AI model ${readiness.ai.model} is configured, but its API key is not set yet. Agent-Reach and Scrapling can stay online; add the Atria API key to PanWatch to enable AI answers.`
    }
    if (!readiness.toolbox.reachable) {
      return `External research is configured but currently unreachable${readiness.toolbox.error ? ` (${readiness.toolbox.error})` : ''}.`
    }
    return null
  }, [readiness])

  const assistantDisabled = disabled || readiness?.ai.api_key_configured === false

  const submit = (nextQuestion = question) => {
    const content = nextQuestion.trim()
    if (!content || assistantDisabled) return
    setQuestion('')
    onSubmit(content)
  }

  return (
    <section className="flex min-h-[calc(100vh-12rem)] flex-1 flex-col items-center justify-center px-5 py-12 text-center md:px-10">
      <p className="mb-4 text-[11px] font-semibold tracking-[0.16em] text-primary sm:text-[12px]">
        PANWATCH · AI INVESTING RESEARCH
      </p>
      <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl md:text-5xl">
        What do you want to research?
      </h1>
      <p className="mt-5 max-w-2xl text-[15px] leading-7 text-muted-foreground md:text-[17px]">
        Ask about an asset, a market question or your portfolio. PanWatch researches available evidence before answering.
      </p>

      {readinessMessage && (
        <div className="mt-6 flex w-full max-w-3xl items-start gap-3 rounded-2xl border border-amber-500/25 bg-amber-500/8 px-4 py-3 text-left">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
          <div>
            <p className="text-[13px] font-medium text-foreground">Assistant setup incomplete</p>
            <p className="mt-1 text-[12px] leading-5 text-muted-foreground">{readinessMessage}</p>
            {readiness?.toolbox.reachable && (
              <p className="mt-1 text-[11px] text-muted-foreground">
                External research gateway: online · {readiness.toolbox.tool_count} tools available
              </p>
            )}
          </div>
        </div>
      )}

      <form
        className="mt-9 flex w-full max-w-3xl items-center gap-2 rounded-2xl border border-border/70 bg-background p-2 shadow-[0_18px_50px_-32px_hsl(var(--foreground)/0.5)] transition-shadow focus-within:ring-2 focus-within:ring-primary/20"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <Search className="ml-3 h-5 w-5 shrink-0 text-muted-foreground" />
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          disabled={assistantDisabled}
          className="h-12 min-w-0 flex-1 bg-transparent text-[15px] text-foreground outline-none placeholder:text-muted-foreground/80"
          placeholder={assistantDisabled ? 'AI API key required before research can run' : 'Ask about XAUUSD, markets, your portfolio, or a research question…'}
          aria-label="Start research"
        />
        <button
          type="submit"
          disabled={assistantDisabled || !question.trim()}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
          aria-label="Send research question"
        >
          <ArrowUpRight className="h-4 w-4" />
        </button>
      </form>

      <div className="mt-7 flex flex-wrap justify-center gap-2.5">
        {QUICK_QUESTIONS.map(({ label, question: quickQuestion, icon: Icon }) => (
          <button
            key={label}
            type="button"
            disabled={assistantDisabled}
            onClick={() => submit(quickQuestion)}
            className="inline-flex items-center gap-2 rounded-xl border border-border/60 bg-card px-4 py-2.5 text-[13px] font-medium text-foreground shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/30 hover:bg-primary/5 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Icon className="h-3.5 w-3.5 text-primary" />
            {label}
          </button>
        ))}
      </div>

      <div className="mt-16 grid w-full max-w-3xl gap-3 text-left sm:grid-cols-3">
        {[
          ['01', 'Start with an asset', 'Enter a symbol or asset name for comprehensive, short-term or event-driven research.'],
          ['02', 'Start with your portfolio', 'Use your live and paper portfolio data to identify concentration and risk exposure.'],
          ['03', 'Start with a question', 'Let the assistant connect price action, technicals and news into the next research steps.'],
        ].map(([index, title, description]) => (
          <div key={index} className="rounded-2xl border border-border/60 bg-card/70 p-5">
            <span className="inline-flex rounded-lg bg-primary/10 px-2 py-1 text-[12px] font-semibold text-primary">{index}</span>
            <h2 className="mt-5 text-[15px] font-semibold text-foreground">{title}</h2>
            <p className="mt-2 text-[12px] leading-5 text-muted-foreground">{description}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
