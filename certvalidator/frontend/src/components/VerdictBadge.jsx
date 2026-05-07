import { ShieldCheck, ShieldX, AlertTriangle, HelpCircle } from 'lucide-react'
import clsx from 'clsx'

const CFG = {
  GENUINE:      { icon: ShieldCheck,    cls: 'bg-green-950/60 text-green-400 border-green-500/30',    label: 'Genuine' },
  FAKE:         { icon: ShieldX,        cls: 'bg-red-950/60 text-red-400 border-red-500/30',          label: 'Fake / Tampered' },
  SUSPICIOUS:   { icon: AlertTriangle,  cls: 'bg-yellow-950/60 text-yellow-400 border-yellow-500/30', label: 'Suspicious' },
  INCONCLUSIVE: { icon: HelpCircle,     cls: 'bg-slate-800/60 text-slate-400 border-slate-500/30',    label: 'Inconclusive' },
}

export function VerdictBadge({ verdict, size = 'md' }) {
  const cfg = CFG[verdict] || CFG.INCONCLUSIVE
  const Icon = cfg.icon
  const px = size === 'lg' ? 'px-4 py-2 text-sm gap-2' : 'px-3 py-1 text-xs gap-1.5'
  return (
    <span className={clsx('inline-flex items-center rounded-full border font-medium', px, cfg.cls)}>
      <Icon size={size === 'lg' ? 16 : 12} />
      {cfg.label}
    </span>
  )
}
