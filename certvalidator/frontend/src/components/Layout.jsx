import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { ShieldCheck, Upload, History, LayoutDashboard, Cpu, Activity } from 'lucide-react'
import clsx from 'clsx'
import { useStats } from '../hooks/useStats'

const NAV = [
  { to: '/',        label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/verify',  label: 'Verify',    icon: Upload },
  { to: '/history', label: 'History',   icon: History },
]

const MODEL_STATUS = [
  { name: 'Preprocessing', live: true  },
  { name: 'Forgery CNN',   live: false },
  { name: 'LayoutLMv3',    live: false },
  { name: 'Mistral-7B',    live: false },
]

export default function Layout() {
  const { data } = useStats()

  return (
    <div className="flex min-h-screen">
      <aside className="w-60 shrink-0 flex flex-col border-r border-white/[0.06] bg-slate-950/90 backdrop-blur-xl sticky top-0 h-screen">
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-white/[0.06]">
          <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/30">
            <ShieldCheck size={16} className="text-white" />
          </div>
          <div>
            <div className="font-semibold text-sm text-white tracking-wide">CertValidator</div>
            <div className="text-xs text-slate-600">AI Forensics</div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => clsx(
              'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150',
              isActive
                ? 'bg-indigo-600/15 text-indigo-400 border border-indigo-500/20'
                : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.04]'
            )}>
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Live counters */}
        {data.total_verifications > 0 && (
          <div className="px-4 mx-3 mb-3 py-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
            <div className="flex items-center gap-2 mb-2">
              <Activity size={11} className="text-slate-500" />
              <span className="text-xs text-slate-500">Live stats</span>
            </div>
            <div className="grid grid-cols-3 gap-1 text-center">
              <div>
                <div className="text-sm font-medium text-green-400">{data.genuine}</div>
                <div className="text-xs text-slate-600">genuine</div>
              </div>
              <div>
                <div className="text-sm font-medium text-red-400">{data.fake}</div>
                <div className="text-xs text-slate-600">fake</div>
              </div>
              <div>
                <div className="text-sm font-medium text-yellow-400">{data.suspicious}</div>
                <div className="text-xs text-slate-600">suspicious</div>
              </div>
            </div>
          </div>
        )}

        {/* Model status */}
        <div className="px-4 mx-3 mb-4 py-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
          <div className="flex items-center gap-2 mb-2">
            <Cpu size={11} className="text-slate-500" />
            <span className="text-xs text-slate-500">Model status</span>
          </div>
          {MODEL_STATUS.map(m => (
            <div key={m.name} className="flex items-center justify-between py-0.5">
              <span className="text-xs text-slate-600">{m.name}</span>
              <div className={clsx('w-1.5 h-1.5 rounded-full', m.live ? 'bg-green-400' : 'bg-yellow-400/60')} />
            </div>
          ))}
          <div className="mt-1.5 text-xs text-slate-700">Green = loaded · Amber = Phase 2</div>
        </div>
      </aside>

      <main className="flex-1 overflow-auto min-h-screen">
        <Outlet />
      </main>
    </div>
  )
}
