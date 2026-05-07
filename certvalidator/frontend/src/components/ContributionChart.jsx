import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts'

const NAMES = {
  forgery_detector: 'Forgery detector',
  field_extractor:  'Field extractor',
  nlp_reasoning:    'NLP reasoning',
  institution_match:'Institution match',
}

export function ContributionChart({ contributions = {}, trustScore = 0 }) {
  const data = Object.entries(contributions)
    .filter(([, v]) => v > 0)
    .map(([key, val]) => ({
      name:  NAMES[key] || key,
      value: Math.round(val * 10) / 10,
      fill:  key === 'forgery_detector' ? '#818cf8'
           : key === 'field_extractor'  ? '#4ade80'
           : key === 'nlp_reasoning'    ? '#fbbf24'
           :                             '#94a3b8',
    }))

  if (data.length === 0) return null

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
      <h3 className="text-sm font-medium text-white mb-4">Score breakdown</h3>

      <div className="flex items-center gap-4">
        <div className="w-28 h-28 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart innerRadius={24} outerRadius={52} data={data} startAngle={90} endAngle={-270}>
              <PolarAngleAxis type="number" domain={[0, 50]} tick={false} />
              <RadialBar dataKey="value" cornerRadius={4} />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>

        <div className="flex-1 space-y-2">
          {data.map(d => (
            <div key={d.name} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full shrink-0" style={{ background: d.fill }} />
                <span className="text-xs text-slate-400">{d.name}</span>
              </div>
              <span className="text-xs font-mono" style={{ color: d.fill }}>+{d.value}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-white/8 flex justify-between items-center">
        <span className="text-xs text-slate-500">Total trust score</span>
        <span className="text-sm font-medium text-white">{trustScore.toFixed(1)} / 100</span>
      </div>
    </div>
  )
}
