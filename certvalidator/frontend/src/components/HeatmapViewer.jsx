import { useState } from 'react'
import { Layers, Download } from 'lucide-react'
import { verify } from '../utils/api'

export function HeatmapViewer({ verificationId, verdict }) {
  const [opacity, setOpacity] = useState(60)
  const [loaded, setLoaded]   = useState(false)
  const [error, setError]     = useState(false)

  const url = verify.heatmap(verificationId)

  const borderColor = {
    GENUINE:    'border-green-500/30',
    FAKE:       'border-red-500/30',
    SUSPICIOUS: 'border-yellow-500/30',
  }[verdict] || 'border-white/10'

  if (error) return null

  return (
    <div className={`rounded-2xl border ${borderColor} bg-white/[0.03] p-5`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-white flex items-center gap-2">
          <Layers size={14} className="text-indigo-400" />
          GradCAM tamper heatmap
        </h3>
        <a href={url} download={`heatmap_${verificationId?.slice(0,8)}.png`}
          className="text-slate-500 hover:text-slate-300 transition-colors p-1">
          <Download size={14} />
        </a>
      </div>

      {/* Opacity slider */}
      <div className="flex items-center gap-3 mb-3">
        <span className="text-xs text-slate-500 w-16 shrink-0">Opacity {opacity}%</span>
        <input type="range" min={0} max={100} value={opacity}
          onChange={e => setOpacity(Number(e.target.value))}
          className="flex-1 accent-indigo-500 h-1" />
      </div>

      {/* Heatmap image */}
      <div className="relative rounded-xl overflow-hidden bg-black/20">
        {!loaded && !error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-5 h-5 rounded-full border-2 border-indigo-500/30 border-t-indigo-400 animate-spin" />
          </div>
        )}
        <img
          src={url}
          alt="GradCAM heatmap"
          style={{ opacity: opacity / 100 }}
          className="w-full h-auto max-h-72 object-contain transition-opacity"
          onLoad={() => setLoaded(true)}
          onError={() => setError(true)}
        />
      </div>

      <p className="text-xs text-slate-600 mt-2">
        Red/yellow = high forgery signal · Blue = low suspicion
      </p>
    </div>
  )
}
