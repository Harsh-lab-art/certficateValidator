import { useNavigate } from 'react-router-dom'
import { Upload, ShieldCheck, ShieldX, AlertTriangle, ArrowRight, Zap, TrendingUp } from 'lucide-react'
import { motion } from 'framer-motion'
import { useStats } from '../hooks/useStats'

const fade = { hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0 } }

const FEATURES = [
  { title: 'ELA forensics',      desc: 'Error Level Analysis reveals JPEG recompression artefacts left by Photoshop edits — invisible to the human eye.', color: 'from-violet-500 to-indigo-600' },
  { title: 'GradCAM heatmap',    desc: 'Pixel-level tamper overlay shows exactly which certificate regions triggered the forgery detector.', color: 'from-teal-500 to-cyan-600' },
  { title: 'NLP reasoning',      desc: 'Mistral-7B cross-checks field consistency, date logic, and issuer patterns for semantic anomalies.', color: 'from-orange-500 to-amber-600' },
  { title: 'Chrome extension',   desc: 'Select any certificate region on screen — the extension captures, verifies, and shows a verdict in seconds.', color: 'from-pink-500 to-rose-600' },
]

const PHASES = [
  { label: 'Phase 1', sub: 'Foundation', done: true  },
  { label: 'Phase 2', sub: 'DL Models',  done: true  },
  { label: 'Phase 3', sub: 'Backend',    done: true  },
  { label: 'Phase 4', sub: 'UI + Ext',   done: true  },
  { label: 'Phase 5', sub: 'Demo',       done: false },
]

export default function HomePage() {
  const navigate = useNavigate()
  const { data, loading } = useStats()

  const stats = [
    { label: 'Verified total', value: data.total_verifications, icon: ShieldCheck, color: 'text-indigo-400' },
    { label: 'Genuine',        value: data.genuine,             icon: ShieldCheck, color: 'text-green-400' },
    { label: 'Fake / tampered',value: data.fake,                icon: ShieldX,     color: 'text-red-400'   },
    { label: 'Suspicious',     value: data.suspicious,          icon: AlertTriangle,color:'text-yellow-400' },
  ]

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <motion.div initial="hidden" animate="show"
        variants={{ show: { transition: { staggerChildren: 0.07 } } }} className="mb-10">
        <motion.div variants={fade} className="flex items-center gap-2 mb-2">
          <Zap size={13} className="text-indigo-400" />
          <span className="text-xs font-medium text-indigo-400 uppercase tracking-widest">AI Forensics Platform</span>
        </motion.div>
        <motion.h1 variants={fade} className="text-3xl font-semibold text-white mb-2">
          Certificate Validator
        </motion.h1>
        <motion.p variants={fade} className="text-slate-400 max-w-lg text-sm leading-relaxed">
          Deep learning pipeline combining EfficientNet + ELA forgery detection, LayoutLMv3 field extraction,
          and Mistral-7B reasoning into a single forensic trust score.
        </motion.p>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-10">
        {stats.map((s, i) => (
          <motion.div key={s.label} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06 }}
            className="rounded-2xl border border-white/[0.06] bg-white/[0.03] p-5">
            <s.icon size={15} className={`${s.color} mb-3`} />
            <div className={`text-2xl font-semibold ${s.color} mb-0.5`}>
              {loading ? '—' : s.value}
            </div>
            <div className="text-xs text-slate-500">{s.label}</div>
          </motion.div>
        ))}
      </div>

      {/* CTA */}
      <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.28 }}
        className="rounded-2xl border border-white/[0.06] bg-white/[0.03] p-7 mb-10 flex flex-col sm:flex-row items-center gap-6">
        <div className="flex-1">
          <h2 className="text-lg font-medium text-white mb-1.5">Verify a certificate now</h2>
          <p className="text-sm text-slate-400">Upload a JPG, PNG, or PDF. The AI pipeline returns a full forensic report.</p>
        </div>
        <button onClick={() => navigate('/verify')}
          className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-6 py-2.5 rounded-xl transition-all active:scale-95 flex items-center gap-2 shrink-0 text-sm">
          <Upload size={14} /> Upload certificate <ArrowRight size={13} />
        </button>
      </motion.div>

      {/* Feature cards */}
      <div className="mb-8">
        <h2 className="text-xs font-medium text-slate-500 uppercase tracking-widest mb-4">Detection pipeline</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {FEATURES.map((f, i) => (
            <motion.div key={f.title} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.32 + i * 0.07 }}
              className="rounded-2xl border border-white/[0.06] bg-white/[0.03] p-5 hover:border-white/[0.12] transition-colors">
              <div className={`w-8 h-0.5 rounded-full bg-gradient-to-r ${f.color} mb-4`} />
              <div className="text-sm font-medium text-white mb-1.5">{f.title}</div>
              <div className="text-xs text-slate-500 leading-relaxed">{f.desc}</div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Phase tracker */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.55 }}
        className="rounded-2xl border border-white/[0.06] bg-white/[0.03] p-5">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp size={12} className="text-slate-500" />
          <span className="text-xs font-medium text-slate-500 uppercase tracking-widest">Build progress</span>
        </div>
        <div className="grid grid-cols-5 gap-2 text-xs">
          {PHASES.map(p => (
            <div key={p.label}
              className={`p-2.5 rounded-xl text-center border ${p.done ? 'border-green-500/30 bg-green-950/40' : 'border-white/[0.06] bg-white/[0.02]'}`}>
              <div className={`font-medium mb-0.5 ${p.done ? 'text-green-400' : 'text-slate-500'}`}>{p.label}</div>
              <div className="text-slate-600">{p.sub}</div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
