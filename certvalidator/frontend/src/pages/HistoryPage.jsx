import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { History, ChevronRight, RefreshCw, Shield } from 'lucide-react'
import { VerdictBadge } from '../components/VerdictBadge'
import { verify } from '../utils/api'
import clsx from 'clsx'

export default function HistoryPage() {
  const navigate = useNavigate()
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)

  const load = async () => {
    try {
      const { data } = await verify.history({ limit: 50 })
      setRecords(data.results || [])
    } catch { setRecords([]) } finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const scoreColor = s =>
    s >= 75 ? 'text-green-400' : s >= 45 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <History size={14} className="text-indigo-400" />
              <span className="text-xs font-medium text-indigo-400 uppercase tracking-widest">History</span>
            </div>
            <h1 className="text-2xl font-semibold text-white">Verification history</h1>
          </div>
          <button onClick={load}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm text-slate-400 bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] transition-all">
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw size={20} className="text-indigo-400 animate-spin" />
          </div>
        ) : records.length === 0 ? (
          <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] p-14 text-center">
            <History size={28} className="text-slate-700 mx-auto mb-3" />
            <p className="text-slate-500 text-sm mb-1">No verifications yet</p>
            <p className="text-slate-600 text-xs mb-5">Verify a certificate to see results here</p>
            <button onClick={() => navigate('/verify')}
              className="inline-flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-500 transition-all">
              <Shield size={14} /> Verify certificate
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            {records.map((r, i) => (
              <motion.div key={r.verification_id || i}
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
                onClick={() => navigate(`/result/${r.verification_id}`, { state: { result: r } })}
                className="group flex items-center gap-4 p-4 rounded-2xl border border-white/[0.06] bg-white/[0.03] hover:border-white/[0.12] hover:bg-white/[0.05] transition-all cursor-pointer">

                {/* Score */}
                <div className={clsx('text-2xl font-semibold w-14 text-right shrink-0', scoreColor(r.trust_score))}>
                  {r.trust_score?.toFixed(0)}
                </div>

                {/* Verdict */}
                <div className="shrink-0">
                  <VerdictBadge verdict={r.verdict} />
                </div>

                {/* Fields */}
                <div className="flex-1 min-w-0">
                  {r.field_scores?.find(f => f.field === 'student_name')?.value && (
                    <p className="text-sm text-slate-300 truncate">
                      {r.field_scores.find(f => f.field === 'student_name').value}
                    </p>
                  )}
                  {r.field_scores?.find(f => f.field === 'institution')?.value && (
                    <p className="text-xs text-slate-500 truncate">
                      {r.field_scores.find(f => f.field === 'institution').value}
                    </p>
                  )}
                </div>

                {/* Meta */}
                <div className="text-right shrink-0">
                  <p className="text-xs text-slate-500">
                    {r.processing_time_s ? `${r.processing_time_s.toFixed(1)}s` : '—'}
                  </p>
                  <p className="text-xs text-slate-700 font-mono">{r.verification_id?.slice(0,8)}</p>
                </div>

                <ChevronRight size={14} className="text-slate-600 group-hover:text-slate-400 transition-colors shrink-0" />
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  )
}
