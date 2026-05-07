import { useEffect, useState } from 'react'
import { useParams, useLocation, useNavigate, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ChevronRight, Download, RefreshCw, Info, Brain, Shield } from 'lucide-react'
import { TrustScoreRing } from '../components/TrustScoreRing'
import { VerdictBadge } from '../components/VerdictBadge'
import { FieldBreakdown } from '../components/FieldBreakdown'
import { HeatmapViewer } from '../components/HeatmapViewer'
import { ContributionChart } from '../components/ContributionChart'
import { verify } from '../utils/api'

export default function ResultPage() {
  const { id } = useParams()
  const location = useLocation()
  const navigate = useNavigate()
  const [result, setResult] = useState(location.state?.result || null)
  const [loading, setLoading] = useState(!result)

  useEffect(() => {
    if (result) return
    let interval
    const poll = async () => {
      try {
        const { data } = await verify.poll(id)
        if (data.status === 'done' || data.status === 'error') {
          setResult(data)
          setLoading(false)
          clearInterval(interval)
        }
      } catch {
        setLoading(false)
        clearInterval(interval)
      }
    }
    poll()
    interval = setInterval(poll, 2000)
    return () => clearInterval(interval)
  }, [id])

  if (loading) return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <RefreshCw size={24} className="text-indigo-400 animate-spin mx-auto mb-3" />
        <p className="text-slate-400 text-sm">Loading result...</p>
      </div>
    </div>
  )

  if (!result) return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <p className="text-slate-400">Result not found.</p>
        <button onClick={() => navigate('/verify')} className="mt-3 text-sm text-indigo-400 hover:underline">
          Verify another
        </button>
      </div>
    </div>
  )

  const { verdict, trust_score, explanation, forgery_score, field_confidence,
          nlp_anomaly_score, institution_matched, field_scores, tamper_regions,
          nlp_reasoning, contributions, confidence_interval, verification_id } = result

  const verdictGlow = {
    GENUINE:    'shadow-green-500/10',
    FAKE:       'shadow-red-500/10',
    SUSPICIOUS: 'shadow-yellow-500/10',
  }[verdict] || ''

  return (
    <div className="p-8 max-w-5xl mx-auto">

      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-6">
        <Link to="/verify" className="hover:text-slate-300 transition-colors">Verify</Link>
        <ChevronRight size={12} />
        <span className="font-mono">{id?.slice(0,8)}...</span>
      </div>

      {/* Hero — verdict + score */}
      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
        className={`rounded-2xl border border-white/[0.06] bg-white/[0.03] p-6 mb-5 shadow-lg ${verdictGlow}`}>
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
          <TrustScoreRing score={trust_score} verdict={verdict} size={148} />

          <div className="flex-1">
            <div className="flex items-center gap-3 mb-3">
              <VerdictBadge verdict={verdict} size="lg" />
              {confidence_interval > 0 && (
                <span className="text-xs text-slate-500">±{confidence_interval.toFixed(1)} CI</span>
              )}
            </div>
            <p className="text-sm text-slate-300 leading-relaxed mb-4 max-w-xl">
              {explanation}
            </p>

            {/* Sub-score pills */}
            <div className="flex flex-wrap gap-2">
              {[
                { label: 'Forgery',     val: `${((1 - (forgery_score || 0)) * 100).toFixed(0)}%`, color: 'text-indigo-400' },
                { label: 'Fields',      val: `${((field_confidence || 0) * 100).toFixed(0)}%`,     color: 'text-green-400' },
                { label: 'NLP',         val: `${((1 - (nlp_anomaly_score || 0)) * 100).toFixed(0)}%`, color: 'text-yellow-400' },
                { label: 'Institution', val: institution_matched ? 'Matched' : 'Unknown',          color: institution_matched ? 'text-green-400' : 'text-red-400' },
              ].map(p => (
                <div key={p.label}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs">
                  <span className="text-slate-500">{p.label}</span>
                  <span className={`font-medium ${p.color}`}>{p.val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Download PDF */}
          <a href={verify.report(verification_id)} download
            className="shrink-0 flex items-center gap-1.5 px-4 py-2 rounded-xl bg-white/[0.04] border border-white/[0.08] text-sm text-slate-400 hover:text-slate-200 hover:bg-white/[0.08] transition-all">
            <Download size={13} /> Export PDF
          </a>
        </div>
      </motion.div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          <FieldBreakdown fieldScores={field_scores} />
        </motion.div>

        <div className="space-y-4">
          <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.12 }}>
            <ContributionChart contributions={contributions} trustScore={trust_score} />
          </motion.div>

          {/* NLP Reasoning */}
          <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.18 }}
            className="rounded-2xl border border-white/[0.06] bg-white/[0.03] p-5">
            <h3 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
              <Brain size={14} className="text-indigo-400" /> AI reasoning
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              {nlp_reasoning || 'Reasoning not available.'}
            </p>

            {tamper_regions?.length > 0 && (
              <div className="mt-3 pt-3 border-t border-white/[0.06]">
                <p className="text-xs font-medium text-red-400 mb-1.5">
                  {tamper_regions.length} tamper region{tamper_regions.length > 1 ? 's' : ''} detected
                </p>
                {tamper_regions.slice(0, 3).map((r, i) => (
                  <p key={i} className="text-xs text-slate-600 font-mono">
                    [{r.x},{r.y}] {r.width}×{r.height}px — {(r.confidence*100).toFixed(0)}%
                  </p>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>

      {/* Heatmap */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.24 }}>
        <HeatmapViewer verificationId={verification_id} verdict={verdict} />
      </motion.div>

      {/* Actions */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
        className="mt-5 flex gap-3 justify-end">
        <button onClick={() => navigate('/history')}
          className="px-4 py-2 rounded-xl text-sm text-slate-400 bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] transition-all">
          View history
        </button>
        <button onClick={() => navigate('/verify')}
          className="px-5 py-2 rounded-xl text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-500 transition-all active:scale-95 flex items-center gap-2">
          <Shield size={14} /> Verify another
        </button>
      </motion.div>
    </div>
  )
}
