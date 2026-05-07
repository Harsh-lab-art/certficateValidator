import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Shield } from 'lucide-react'
import { DropZone } from '../components/DropZone'
import { ProcessingSteps } from '../components/ProcessingSteps'
import { useVerification } from '../hooks/useVerification'

export default function VerifyPage() {
  const navigate = useNavigate()
  const { status, steps, stepIdx, result, error, submit, reset } = useVerification()

  // Navigate to result when done
  if (status === 'done' && result) {
    navigate(`/result/${result.verification_id}`, { state: { result } })
  }

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2 mb-1">
          <Shield size={14} className="text-indigo-400" />
          <span className="text-xs font-medium text-indigo-400 uppercase tracking-widest">Verification</span>
        </div>
        <h1 className="text-2xl font-semibold text-white mb-1">Verify a certificate</h1>
        <p className="text-slate-400 text-sm mb-8">
          Upload the certificate image or PDF. The AI pipeline runs automatically and returns a forensic report.
        </p>

        {/* Upload or processing */}
        {status === 'idle' || status === 'error' ? (
          <>
            <DropZone onFile={submit} disabled={false} />
            {status === 'error' && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="mt-4 p-4 rounded-xl bg-red-950/40 border border-red-500/20">
                <p className="text-sm text-red-400">{error}</p>
                <button onClick={reset} className="text-xs text-red-400/70 hover:text-red-400 mt-2 transition-colors">
                  Try again
                </button>
              </motion.div>
            )}
          </>
        ) : (
          <ProcessingSteps steps={steps} currentIdx={stepIdx} />
        )}

        {/* Tips */}
        <div className="mt-8 grid grid-cols-3 gap-3">
          {[
            { title: 'Best results',   body: 'Use high-res scans at 300 DPI or above' },
            { title: 'Formats',        body: 'JPG, PNG, TIFF, or PDF (first page)' },
            { title: 'Privacy',        body: 'Files are not stored beyond the session' },
          ].map(t => (
            <div key={t.title}
              className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
              <div className="text-xs font-medium text-slate-400 mb-1">{t.title}</div>
              <div className="text-xs text-slate-600 leading-relaxed">{t.body}</div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
