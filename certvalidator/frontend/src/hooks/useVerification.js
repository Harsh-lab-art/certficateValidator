import { useState, useRef, useCallback } from 'react'
import { verify } from '../utils/api'
import toast from 'react-hot-toast'

export function useVerification() {
  const [state, setState] = useState({
    status: 'idle',      // idle | uploading | processing | done | error
    stepIdx: 0,
    result: null,
    verificationId: null,
    error: null,
  })
  const pollRef = useRef(null)

  const STEPS = [
    { id: 'upload',     label: 'Uploading certificate...' },
    { id: 'preprocess', label: 'Deskewing · denoising · normalising...' },
    { id: 'ela',        label: 'Computing ELA forensic channel...' },
    { id: 'ocr',        label: 'Extracting text with TrOCR...' },
    { id: 'forgery',    label: 'Running forgery detector (EfficientNet+ELA)...' },
    { id: 'layout',     label: 'Extracting fields with LayoutLMv3...' },
    { id: 'nlp',        label: 'NLP reasoning with Mistral-7B...' },
    { id: 'fusion',     label: 'Computing trust score...' },
  ]

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const submit = useCallback(async (file) => {
    stopPolling()
    setState({ status: 'uploading', stepIdx: 0, result: null, verificationId: null, error: null })

    const form = new FormData()
    form.append('file', file)

    try {
      const { data } = await verify.submit(form)
      const vid = data.verification_id

      setState(s => ({ ...s, status: 'processing', stepIdx: 1, verificationId: vid }))

      // Step ticker — advances UI while we wait
      let tick = 1
      const ticker = setInterval(() => {
        tick = Math.min(tick + 1, STEPS.length - 1)
        setState(s => ({ ...s, stepIdx: tick }))
      }, 1800)

      // Poll for result
      pollRef.current = setInterval(async () => {
        try {
          const { data: res } = await verify.poll(vid)
          if (res.status === 'done') {
            clearInterval(ticker)
            stopPolling()
            setState({
              status: 'done',
              stepIdx: STEPS.length - 1,
              result: res,
              verificationId: vid,
              error: null,
            })
            toast.success(`Verification complete — ${res.verdict}`)
          } else if (res.status === 'error') {
            clearInterval(ticker)
            stopPolling()
            setState(s => ({ ...s, status: 'error', error: res.detail || 'Verification failed' }))
            toast.error('Verification failed')
          }
        } catch {
          // keep polling on transient network errors
        }
      }, 2000)

    } catch (err) {
      const msg = err.response?.data?.detail || 'Upload failed. Is the backend running?'
      setState(s => ({ ...s, status: 'error', error: msg }))
      toast.error(msg)
    }
  }, [])

  const reset = useCallback(() => {
    stopPolling()
    setState({ status: 'idle', stepIdx: 0, result: null, verificationId: null, error: null })
  }, [])

  return { ...state, steps: STEPS, submit, reset }
}
