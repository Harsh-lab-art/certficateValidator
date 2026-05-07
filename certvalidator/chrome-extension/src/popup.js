const API = 'http://localhost:8000'
const DASH = 'http://localhost:5173'

const $ = id => document.getElementById(id)

// ── API health ────────────────────────────────────────────────────────────
async function checkAPI() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) })
    const d = await r.json()
    if (d.status === 'ok') {
      $('api-dot').className  = 'dot on'
      $('api-text').textContent = `Backend connected · v${d.version}`
      $('btn-scan').disabled  = false
    } else throw new Error()
  } catch {
    $('api-dot').className  = 'dot off'
    $('api-text').textContent = 'Backend offline — run uvicorn'
    $('btn-scan').disabled  = true
  }
}

// ── Scan button ───────────────────────────────────────────────────────────
$('btn-scan').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
  window.close()
  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.__certval_startSelection?.(),
  })
})

// ── Messages from background ──────────────────────────────────────────────
chrome.runtime.onMessage.addListener(msg => {
  if (msg.type === 'VERIFY_RESULT') showResult(msg.data)
  if (msg.type === 'VERIFY_ERROR')  showError(msg.error)
  if (msg.type === 'VERIFY_STEP')   setLoadingStep(msg.step)
})

// ── Pending results (popup re-opened after scan) ──────────────────────────
chrome.storage.local.get(['cvResult','cvError'], ({ cvResult, cvError }) => {
  if (cvResult)  { showResult(cvResult); chrome.storage.local.remove('cvResult') }
  if (cvError)   { showError(cvError);   chrome.storage.local.remove('cvError')  }
})

// ── Loading state ─────────────────────────────────────────────────────────
function setLoadingStep(text) {
  $('loading').classList.add('show')
  $('result').classList.remove('show')
  $('loading-step').textContent = text
}

// ── Show result ───────────────────────────────────────────────────────────
function showResult(data) {
  $('loading').classList.remove('show')
  $('result').classList.add('show')

  const verdict = data.verdict || 'INCONCLUSIVE'
  const score   = Math.round(data.trust_score ?? 0)
  const cls     = verdict.toLowerCase() === 'fake' ? 'fake'
                : verdict.toLowerCase() === 'genuine' ? 'genuine'
                : verdict.toLowerCase() === 'suspicious' ? 'suspicious'
                : 'inconclusive'

  // Verdict card
  $('verdict-card').className = `verdict-card ${cls}`
  $('verdict-text').className = `verdict-text ${cls}`
  $('verdict-text').textContent = verdict === 'FAKE' ? 'Fake / Tampered' : verdict.charAt(0) + verdict.slice(1).toLowerCase()

  // Score ring animation
  const CIRC  = 125.6
  const offset= CIRC - (score / 100) * CIRC
  const col   = cls === 'genuine' ? '#4ade80' : cls === 'fake' ? '#f87171' : cls === 'suspicious' ? '#fbbf24' : '#94a3b8'
  const ring  = $('ring-fill')
  ring.setAttribute('stroke', col)
  $('score-num').style.color = col
  $('score-num').textContent = score

  // Animate ring
  ring.style.transition = 'none'
  ring.setAttribute('stroke-dashoffset', CIRC)
  requestAnimationFrame(() => {
    ring.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.34,1.56,0.64,1)'
    ring.setAttribute('stroke-dashoffset', offset)
  })

  // Sub-score bars
  const fg = Math.round((1 - (data.forgery_score ?? 0.5)) * 100)
  const fc = Math.round((data.field_confidence ?? 0.5) * 100)
  const nlp= Math.round((1 - (data.nlp_anomaly_score ?? 0.5)) * 100)

  animateBar('bar-forgery', fg, '#818cf8')
  animateBar('bar-fields',  fc, '#4ade80')
  animateBar('bar-nlp',     nlp,'#fbbf24')
  $('bar-forgery-val').textContent = fg + '%'
  $('bar-fields-val').textContent  = fc + '%'
  $('bar-nlp-val').textContent     = nlp + '%'

  // Fields
  const container = $('fields')
  container.innerHTML = ''
  const FIELD_LABELS = {
    student_name: 'Name', institution: 'Institution', degree: 'Degree',
    issue_date:   'Date', grade: 'Grade', roll_number: 'Roll no.',
  }
  for (const f of (data.field_scores || [])) {
    const row = document.createElement('div')
    row.className = 'field-row'
    const label = FIELD_LABELS[f.field] || f.field
    row.innerHTML = `
      <span class="field-name">${label}</span>
      <span class="field-val">${f.value || '—'}</span>
      <div class="field-dot ${f.flagged ? 'bad' : 'ok'}"></div>
    `
    container.appendChild(row)
  }

  // NLP reasoning
  if (data.nlp_reasoning) {
    $('reason-box').style.display = 'block'
    $('reason-box').textContent = data.nlp_reasoning
  }

  // Explain strip
  const ci = data.confidence_interval
  $('explain-text').textContent = ci ? `Confidence interval ±${ci.toFixed(1)}` : ''

  // Full report link
  $('view-full').href = `${DASH}/result/${data.verification_id}`
}

function animateBar(id, pct, color) {
  const el = $(id)
  el.style.background = color
  el.style.width = '0%'
  requestAnimationFrame(() => {
    el.style.width = pct + '%'
  })
}

function showError(msg) {
  $('loading').classList.remove('show')
  $('result').classList.add('show')
  $('verdict-card').className = 'verdict-card fake'
  $('verdict-text').className = 'verdict-text fake'
  $('verdict-text').textContent = 'Error'
  $('score-num').textContent = '—'
  $('fields').innerHTML = `<p style="font-size:12px;color:#f87171;padding:8px 0">${msg}</p>`
}

// ── Init ──────────────────────────────────────────────────────────────────
checkAPI()
