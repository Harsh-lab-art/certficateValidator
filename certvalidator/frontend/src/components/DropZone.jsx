import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, FileImage, X, CheckCircle2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import clsx from 'clsx'
import toast from 'react-hot-toast'

const ACCEPTED = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png':  ['.png'],
  'image/tiff': ['.tiff', '.tif'],
  'application/pdf': ['.pdf'],
}

export function DropZone({ onFile, disabled }) {
  const [file, setFile]     = useState(null)
  const [preview, setPreview] = useState(null)

  const onDrop = useCallback(accepted => {
    const f = accepted[0]
    if (!f) return
    setFile(f)
    if (f.type.startsWith('image/')) setPreview(URL.createObjectURL(f))
    else setPreview(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: ACCEPTED, maxSize: 20 * 1024 * 1024, multiple: false,
    disabled,
    onDropRejected: ([r]) => {
      if (r.errors[0]?.code === 'file-too-large') toast.error('File exceeds 20 MB.')
      else toast.error('Unsupported type. Use JPG, PNG, TIFF or PDF.')
    },
  })

  const clear = () => { setFile(null); setPreview(null) }

  return (
    <div className="space-y-3">
      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div key="dz" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            {...getRootProps()}
            className={clsx(
              'border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200',
              isDragActive  ? 'border-indigo-500 bg-indigo-500/10' : 'border-white/10 hover:border-white/20 hover:bg-white/[0.02]',
              disabled && 'opacity-50 cursor-not-allowed pointer-events-none'
            )}
          >
            <input {...getInputProps()} />
            <div className={clsx('w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center transition-colors',
              isDragActive ? 'bg-indigo-500/20' : 'bg-white/5')}>
              <Upload size={26} className={isDragActive ? 'text-indigo-400' : 'text-slate-500'} />
            </div>
            {isDragActive ? (
              <p className="text-indigo-400 font-medium">Drop it here</p>
            ) : (
              <>
                <p className="text-slate-300 font-medium mb-1">Drag & drop certificate</p>
                <p className="text-slate-500 text-sm mb-4">or click to browse files</p>
                <div className="flex items-center justify-center gap-2 flex-wrap">
                  {['JPG', 'PNG', 'TIFF', 'PDF'].map(f => (
                    <span key={f} className="px-2 py-0.5 rounded-md bg-white/5 text-slate-400 text-xs font-mono border border-white/8">{f}</span>
                  ))}
                  <span className="text-slate-600 text-xs">· max 20 MB</span>
                </div>
              </>
            )}
          </motion.div>
        ) : (
          <motion.div key="prev" initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
            className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 flex items-start gap-4">
            <div className="w-20 h-20 rounded-xl bg-white/5 border border-white/10 overflow-hidden shrink-0 flex items-center justify-center">
              {preview
                ? <img src={preview} alt="" className="w-full h-full object-cover" />
                : <FileText size={28} className="text-slate-500" />
              }
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <FileImage size={13} className="text-indigo-400 shrink-0" />
                <span className="text-sm font-medium text-white truncate">{file.name}</span>
              </div>
              <p className="text-xs text-slate-500 mb-3">{(file.size/1024).toFixed(1)} KB · {file.type}</p>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-green-950/60 border border-green-500/30 text-green-400 text-xs font-medium">
                <CheckCircle2 size={11} /> Ready to verify
              </span>
            </div>
            <button onClick={clear} disabled={disabled}
              className="text-slate-500 hover:text-slate-300 transition-colors p-1 shrink-0">
              <X size={15} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {file && !disabled && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
          className="flex justify-end gap-3">
          <button onClick={clear}
            className="px-5 py-2 rounded-xl text-sm font-medium text-slate-400 bg-white/5 hover:bg-white/10 border border-white/10 transition-all">
            Clear
          </button>
          <button onClick={() => onFile(file)}
            className="px-6 py-2 rounded-xl text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-500 transition-all active:scale-95 flex items-center gap-2">
            <Upload size={14} /> Run verification
          </button>
        </motion.div>
      )}
    </div>
  )
}
