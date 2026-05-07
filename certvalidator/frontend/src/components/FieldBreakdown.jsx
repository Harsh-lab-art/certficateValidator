import { motion } from 'framer-motion'
import { Layers, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

function ConfBar({ value, flagged, delay = 0 }) {
  const pct   = Math.round(value * 100)
  const color = flagged       ? 'bg-red-500'
              : pct > 80      ? 'bg-green-500'
              : pct > 60      ? 'bg-yellow-500'
              :                 'bg-red-500'
  return (
    <div className="h-1 flex-1 rounded-full bg-white/8 overflow-hidden">
      <motion.div className={clsx('h-full rounded-full', color)}
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 1, ease: 'easeOut', delay }}
      />
    </div>
  )
}

export function FieldBreakdown({ fieldScores = [] }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
      <h3 className="text-sm font-medium text-white mb-4 flex items-center gap-2">
        <Layers size={14} className="text-indigo-400" />
        Field extraction
      </h3>

      <div className="space-y-3">
        {fieldScores.map((f, i) => (
          <div key={f.field}>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400 capitalize">
                  {f.field.replace(/_/g, ' ')}
                </span>
                {f.flagged && (
                  <span className="flex items-center gap-0.5 text-xs px-1.5 py-0.5 rounded-md bg-red-950/60 text-red-400 border border-red-500/20">
                    <AlertCircle size={9} /> flagged
                  </span>
                )}
              </div>
              <span className="text-xs font-mono text-slate-500">
                {Math.round(f.confidence * 100)}%
              </span>
            </div>
            <div className="flex items-center gap-3">
              <ConfBar value={f.confidence} flagged={f.flagged} delay={i * 0.08} />
            </div>
            {f.value && (
              <p className="text-xs text-slate-600 mt-0.5 font-mono truncate">{f.value}</p>
            )}
            {f.issues?.length > 0 && (
              <p className="text-xs text-red-400/70 mt-0.5">{f.issues[0]}</p>
            )}
          </div>
        ))}

        {fieldScores.length === 0 && (
          <p className="text-xs text-slate-500 text-center py-4">
            No field data available — backend may still be processing.
          </p>
        )}
      </div>
    </div>
  )
}
