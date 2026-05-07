;(function () {
  if (window.__certval_init) return
  window.__certval_init = true

  const API = 'http://localhost:8000'

  let overlay, selBox, isDrawing = false, sx = 0, sy = 0

  // ── Public trigger ────────────────────────────────────────────────────────
  window.__certval_startSelection = () => {
    if (document.getElementById('__cv_overlay')) return
    buildOverlay()
  }

  // ── Selection overlay ─────────────────────────────────────────────────────
  function buildOverlay() {
    overlay = document.createElement('div')
    overlay.id = '__cv_overlay'
    overlay.style.cssText = `
      position:fixed;inset:0;z-index:2147483647;cursor:crosshair;
      background:rgba(0,0,0,0.3);
    `

    const banner = document.createElement('div')
    banner.style.cssText = `
      position:absolute;top:16px;left:50%;transform:translateX(-50%);
      background:#0f172a;color:#e2e8f0;padding:8px 18px;border-radius:8px;
      font:500 12px/1.4 -apple-system,sans-serif;
      border:1px solid rgba(99,102,241,0.4);pointer-events:none;white-space:nowrap;
      box-shadow:0 4px 20px rgba(0,0,0,0.5);
    `
    banner.textContent = 'Drag to select certificate region · Esc to cancel'
    overlay.appendChild(banner)

    selBox = document.createElement('div')
    selBox.style.cssText = `
      position:absolute;border:2px solid #6366f1;
      background:rgba(99,102,241,0.1);border-radius:4px;
      display:none;pointer-events:none;
      box-shadow:0 0 0 1px rgba(99,102,241,0.3);
    `
    overlay.appendChild(selBox)

    overlay.addEventListener('mousedown', onDown)
    overlay.addEventListener('mousemove', onMove)
    overlay.addEventListener('mouseup',   onUp)
    document.addEventListener('keydown',  onEsc)
    document.body.appendChild(overlay)
  }

  function removeOverlay() {
    overlay?.remove(); overlay = null
    document.removeEventListener('keydown', onEsc)
  }

  function onDown(e) {
    isDrawing = true; sx = e.clientX; sy = e.clientY
    selBox.style.display = 'block'
    updateBox(e.clientX, e.clientY)
  }

  function onMove(e) {
    if (!isDrawing) return
    updateBox(e.clientX, e.clientY)
  }

  async function onUp(e) {
    if (!isDrawing) return
    isDrawing = false
    const w = Math.abs(e.clientX - sx), h = Math.abs(e.clientY - sy)
    const x = Math.min(sx, e.clientX), y = Math.min(sy, e.clientY)
    removeOverlay()
    if (w < 40 || h < 40) { showToast('Selection too small — try again', 'warn'); return }
    await captureAndVerify(x, y, w, h)
  }

  function onEsc(e) { if (e.key === 'Escape') removeOverlay() }

  function updateBox(x, y) {
    const l = Math.min(sx,x), t = Math.min(sy,y)
    const w = Math.abs(x-sx),  h = Math.abs(y-sy)
    selBox.style.left=l+'px'; selBox.style.top=t+'px'
    selBox.style.width=w+'px'; selBox.style.height=h+'px'
  }

  // ── Capture → API ─────────────────────────────────────────────────────────
  async function captureAndVerify(x, y, w, h) {
    const toast = showToast('Capturing region...')
    notifyStep('Capturing region...')

    try {
      const dataUrl = await chrome.runtime.sendMessage({ type: 'CAPTURE_TAB' })
      if (!dataUrl) throw new Error('Tab capture failed')

      toast.setText('Cropping to selected area...')
      notifyStep('Cropping to selected area...')
      const dpr  = window.devicePixelRatio || 1
      const blob = await cropDataUrl(dataUrl, x*dpr, y*dpr, w*dpr, h*dpr)

      toast.setText('Uploading to CertValidator...')
      notifyStep('Uploading certificate...')
      const form = new FormData()
      form.append('file', blob, 'screen_capture.jpg')

      const res = await fetch(`${API}/api/v1/verify`, { method:'POST', body:form })
      if (!res.ok) throw new Error(`API ${res.status}`)
      const submitted = await res.json()

      toast.setText('Running AI pipeline...')
      notifyStep('Running AI forensic pipeline...')
      const result = await pollResult(submitted.verification_id, toast)

      toast.remove()
      await chrome.storage.local.set({ cvResult: result })
      chrome.runtime.sendMessage({ type: 'OPEN_POPUP' })
      showInlineBadge(x, y, w, h, result)

    } catch(err) {
      toast.remove()
      await chrome.storage.local.set({ cvError: err.message })
      chrome.runtime.sendMessage({ type: 'OPEN_POPUP' })
    }
  }

  async function cropDataUrl(dataUrl, x, y, w, h) {
    return new Promise(res => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement('canvas')
        canvas.width=w; canvas.height=h
        canvas.getContext('2d').drawImage(img, x, y, w, h, 0, 0, w, h)
        canvas.toBlob(blob => res(blob), 'image/jpeg', 0.95)
      }
      img.src = dataUrl
    })
  }

  async function pollResult(id, toast, max=20) {
    for (let i=0; i<max; i++) {
      await new Promise(r => setTimeout(r, 2000))
      const r = await fetch(`${API}/api/v1/verify/${id}`)
      const d = await r.json()
      if (d.status === 'done' || d.status === 'error') return d
      // Update step hint
      const steps = ['Preprocessing image...','Running ELA forensics...','Extracting fields...','NLP reasoning...','Computing trust score...']
      if (toast) toast.setText(steps[Math.min(i, steps.length-1)])
      notifyStep(steps[Math.min(i, steps.length-1)])
    }
    throw new Error('Verification timed out')
  }

  function notifyStep(step) {
    chrome.runtime.sendMessage({ type: 'VERIFY_STEP', step })
  }

  // ── Inline badge (shown on page) ──────────────────────────────────────────
  function showInlineBadge(x, y, w, h, result) {
    const verdict = result.verdict || 'INCONCLUSIVE'
    const score   = Math.round(result.trust_score ?? 0)
    const col     = verdict==='GENUINE'?'#4ade80':verdict==='FAKE'?'#f87171':'#fbbf24'

    const badge = document.createElement('div')
    badge.style.cssText = `
      position:fixed;left:${x}px;top:${y+h+10}px;
      background:#0f172a;border:1px solid ${col}55;border-radius:10px;
      padding:9px 14px;display:flex;align-items:center;gap:10px;
      font:500 13px/1 -apple-system,sans-serif;color:#e2e8f0;
      box-shadow:0 4px 24px rgba(0,0,0,0.6);z-index:2147483646;
    `
    badge.innerHTML = `
      <div style="width:8px;height:8px;border-radius:50%;background:${col};flex-shrink:0"></div>
      <span style="color:${col};font-weight:600">${verdict}</span>
      <span style="color:#475569">·</span>
      <span>Score <strong style="color:${col}">${score}</strong>/100</span>
      <button onclick="this.parentElement.remove()" style="
        margin-left:4px;background:none;border:none;color:#475569;
        cursor:pointer;font-size:16px;line-height:1;padding:0
      ">×</button>
    `
    document.body.appendChild(badge)
    setTimeout(() => badge.remove(), 10000)
  }

  // ── Toast ─────────────────────────────────────────────────────────────────
  function showToast(text, type='info') {
    injectStyles()
    const el = document.createElement('div')
    el.className = '__cv_toast'
    el.innerHTML = `
      <div class="__cv_spinner"></div>
      <span id="__cv_step">${text}</span>
    `
    document.body.appendChild(el)
    return {
      setText: t => { const s=el.querySelector('#__cv_step'); if(s) s.textContent=t },
      remove:  () => el.remove(),
    }
  }

  function injectStyles() {
    if (document.getElementById('__cv_styles')) return
    const s = document.createElement('style')
    s.id = '__cv_styles'
    s.textContent = `
      .__cv_toast {
        position:fixed;bottom:24px;right:24px;
        background:#0f172a;border:1px solid rgba(99,102,241,0.35);
        border-radius:10px;padding:10px 15px;
        display:flex;align-items:center;gap:10px;
        font:500 13px/1 -apple-system,sans-serif;color:#e2e8f0;
        box-shadow:0 4px 20px rgba(0,0,0,0.5);z-index:2147483646;
        animation:__cv_in 0.2s ease;
      }
      .__cv_spinner {
        width:14px;height:14px;border-radius:50%;
        border:2px solid rgba(99,102,241,0.3);border-top-color:#6366f1;
        animation:__cv_spin 0.75s linear infinite;flex-shrink:0;
      }
      @keyframes __cv_spin { to{transform:rotate(360deg)} }
      @keyframes __cv_in   { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:none} }
    `
    document.head.appendChild(s)
  }
})()
