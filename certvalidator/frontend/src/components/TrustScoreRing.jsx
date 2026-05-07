import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

const COLORS = {
  GENUINE:    '#4ade80',
  SUSPICIOUS: '#fbbf24',
  FAKE:       '#f87171',
  default:    '#818cf8',
}

const R = 52
const CIRC = 2 * Math.PI * R

export function TrustScoreRing({ score = 0, verdict = 'default', size = 148 }) {
  const filled  = (score / 100) * CIRC
  const offset  = CIRC - filled
  const color   = COLORS[verdict] || COLORS.default

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox="0 0 148 148" className="absolute inset-0 rotate-[-90deg]">
        {/* Track */}
        <circle cx="74" cy="74" r={R} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="10" />
        {/* Filled arc */}
        <motion.circle
          cx="74" cy="74" r={R}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={CIRC}
          initial={{ strokeDashoffset: CIRC }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.6, ease: [0.34, 1.56, 0.64, 1] }}
        />
      </svg>
      {/* Centre text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          className="text-3xl font-semibold leading-none"
          style={{ color }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          {Math.round(score)}
        </motion.span>
        <span className="text-xs text-slate-500 mt-0.5">/ 100</span>
      </div>
    </div>
  )
}
