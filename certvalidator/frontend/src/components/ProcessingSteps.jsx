import { motion } from 'framer-motion'
import { CheckCircle2, Loader2, Circle } from 'lucide-react'
import clsx from 'clsx'

export function ProcessingSteps({ steps, currentIdx }) {
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
      <div className="flex items-center gap-3 mb-5">
        <div className="relative w-7 h-7 flex items-center justify-center">
          <div className="absolute inset-0 rounded-full border-2 border-indigo-500/20" />
          <Loader2 size={14} className="text-indigo-400 animate-spin" />
        </div>
        <span className="text-sm font-medium text-white">Analysing certificate...</span>
      </div>

      <div className="space-y-1.5">
        {steps.map((step, i) => {
          const done   = i < currentIdx
          const active = i === currentIdx
          const future = i > currentIdx
          return (
            <motion.div key={step.id}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: future ? 0.4 : 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className={clsx(
                'flex items-center gap-3 px-3 py-2 rounded-xl text-xs transition-all duration-300',
                active && 'bg-white/5'
              )}>
              {done   && <CheckCircle2 size={13} className="text-green-400 shrink-0" />}
              {active && <Loader2 size={13} className="text-indigo-400 animate-spin shrink-0" />}
              {future && <Circle size={13} className="text-slate-700 shrink-0" />}
              <span className={clsx(
                done   && 'text-slate-500',
                active && 'text-white',
                future && 'text-slate-700',
              )}>{step.label}</span>
            </motion.div>
          )
        })}
      </div>
    </motion.div>
  )
}
