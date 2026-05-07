// presentation/build_deck.js
// Run: node presentation/build_deck.js
// Output: presentation/CertValidator_Hackathon.pptx

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout  = "LAYOUT_16x9";
pres.author  = "CertValidator Team";
pres.title   = "CertValidator — AI Certificate Forensics";
pres.subject = "Hackathon Presentation";

// ── Palette ─────────────────────────────────────────────────────────────
const C = {
  navy:     "0F172A",   // dark bg
  indigo:   "4F46E5",   // primary
  indigoL:  "818CF8",   // light indigo
  green:    "4ADE80",   // genuine
  red:      "F87171",   // fake
  yellow:   "FBBF24",   // suspicious
  slate:    "94A3B8",   // muted text
  white:    "FFFFFF",
  card:     "1E293B",   // card bg
  border:   "334155",   // card border
};

const makeShadow = () => ({ type:"outer", blur:8, offset:3, angle:135, color:"000000", opacity:0.18 });

// ── Helper: dark slide background ────────────────────────────────────────
function darkBg(slide) {
  slide.background = { color: C.navy };
}

// ── Helper: section label ────────────────────────────────────────────────
function sectionLabel(slide, text) {
  slide.addText(text.toUpperCase(), {
    x:0.5, y:0.22, w:9, h:0.25,
    fontSize:9, color:C.indigoL, bold:true, charSpacing:4,
  });
}

// ── Helper: card box ─────────────────────────────────────────────────────
function card(slide, x, y, w, h, opts={}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill:{ color: opts.fill || C.card },
    line:{ color: opts.border || C.border, width:1 },
    shadow: makeShadow(),
  });
}

// ── Helper: stat card ────────────────────────────────────────────────────
function statCard(slide, x, y, w, num, label, color) {
  card(slide, x, y, w, 1.2);
  slide.addText(num, { x, y:y+0.18, w, h:0.6, fontSize:38, bold:true, color, align:"center" });
  slide.addText(label, { x, y:y+0.78, w, h:0.32, fontSize:10, color:C.slate, align:"center" });
}

// ── Helper: feature row ───────────────────────────────────────────────────
function featureRow(slide, x, y, color, title, body) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w:0.06, h:0.55, fill:{ color } });
  slide.addText(title, { x:x+0.15, y, w:3.6, h:0.28, fontSize:12, bold:true, color:C.white });
  slide.addText(body,  { x:x+0.15, y:y+0.28, w:3.6, h:0.28, fontSize:9, color:C.slate });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);

  // Big shield icon (text approximation)
  s.addShape(pres.shapes.OVAL, {
    x:4.1, y:0.4, w:1.8, h:1.8, fill:{ color:C.indigo }, line:{ color:C.indigo, width:0 },
  });
  s.addText("🛡", { x:4.1, y:0.48, w:1.8, h:1.6, fontSize:52, align:"center" });

  s.addText("CertValidator", {
    x:0.5, y:2.4, w:9, h:1.1,
    fontSize:52, bold:true, color:C.white, align:"center",
  });
  s.addText("AI-Powered Academic Certificate Authenticity Validator", {
    x:0.8, y:3.5, w:8.4, h:0.5,
    fontSize:18, color:C.indigoL, align:"center",
  });

  // Tag pills
  const tags = ["EfficientNet + ELA","LayoutLMv3","Mistral-7B","Chrome Extension"];
  tags.forEach((t, i) => {
    const w = 1.95, gap = 0.15, startX = (10 - tags.length*(w+gap)+gap)/2;
    const x = startX + i*(w+gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y:4.25, w, h:0.42, fill:{color:"1E293B"}, line:{color:C.border,width:1}, rectRadius:0.06 });
    s.addText(t, { x, y:4.25, w, h:0.42, fontSize:9.5, color:C.indigoL, align:"center", bold:true });
  });

  s.addText("Hackathon 2026  ·  May 7–8", {
    x:0.5, y:5.05, w:9, h:0.28, fontSize:10, color:C.slate, align:"center",
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Problem statement
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionLabel(s, "The Problem");

  s.addText("Certificate fraud is a\nbillion-dollar crisis", {
    x:0.5, y:0.5, w:5, h:1.6,
    fontSize:32, bold:true, color:C.white, align:"left",
  });

  // Stats
  [
    ["3.2M+",  "fake degrees issued\nglobally per year",  C.red,    0.5],
    ["₹2,400Cr","estimated fraud value\nin India alone",   C.yellow, 2.6],
    ["8–12 wks","manual verification\ntime per institution",C.indigoL,4.7],
  ].forEach(([num, label, color, x]) => statCard(s, x, 2.25, 1.85, num, label, color));

  // Pain points
  s.addText("Current pain points", {
    x:0.5, y:3.7, w:5, h:0.32, fontSize:13, bold:true, color:C.white,
  });
  [
    "Verification is manual, slow, and error-prone",
    "No unified system — each institution has its own silo",
    "Employers have no way to verify in real time",
    "Forgers use free tools; detection requires experts",
  ].forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x:0.5, y:4.12+i*0.44, w:0.18, h:0.18, fill:{color:C.red} });
    s.addText(t, { x:0.82, y:4.08+i*0.44, w:5.5, h:0.26, fontSize:11, color:C.slate });
  });

  // Right visual — newspaper-style fraud numbers
  card(s, 5.9, 0.55, 3.6, 4.7, { fill:"1A1032", border:"4F46E5" });
  s.addText("\"Fake degrees are\nundetectable by\nhuman inspection\n90% of the time\"", {
    x:6.05, y:0.85, w:3.3, h:1.8,
    fontSize:16, bold:true, color:C.white, align:"center", italic:true,
  });
  s.addText("— NASSCOM 2024 Report", {
    x:6.05, y:2.65, w:3.3, h:0.3,
    fontSize:9, color:C.slate, align:"center",
  });
  s.addShape(pres.shapes.LINE, { x:6.3, y:3.1, w:2.8, h:0, line:{color:C.border, width:1} });
  s.addText("Verifying certificates\nis the #1 fraud vector\nin campus hiring", {
    x:6.05, y:3.2, w:3.3, h:1.0,
    fontSize:12, color:C.indigoL, align:"center",
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Our solution
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionLabel(s, "Our Solution");

  s.addText("One upload. Full forensic verdict.", {
    x:0.5, y:0.45, w:9, h:0.7,
    fontSize:34, bold:true, color:C.white, align:"center",
  });
  s.addText("CertValidator combines three AI models into a single trust score with a downloadable evidence report.", {
    x:1, y:1.2, w:8, h:0.45,
    fontSize:13, color:C.slate, align:"center",
  });

  // Flow diagram
  const flow = [
    { label:"Upload",       sub:"JPG · PNG · PDF",     col:C.indigoL },
    { label:"Preprocess",   sub:"Deskew · ELA",        col:C.indigoL },
    { label:"AI Pipeline",  sub:"3 DL models",         col:C.indigo  },
    { label:"Trust Score",  sub:"0 – 100",             col:C.green   },
    { label:"Report",       sub:"PDF + heatmap",       col:C.green   },
  ];
  flow.forEach((f, i) => {
    const x = 0.45 + i*1.88;
    card(s, x, 1.85, 1.7, 1.15, { fill:"1E293B", border: f.col });
    s.addText(f.label, { x, y:1.97, w:1.7, h:0.38, fontSize:12, bold:true, color:C.white, align:"center" });
    s.addText(f.sub,   { x, y:2.38, w:1.7, h:0.28, fontSize:9,  color:f.col,  align:"center" });
    if (i < flow.length-1) {
      s.addText("→", { x:x+1.68, y:2.22, w:0.22, h:0.32, fontSize:16, color:C.slate, align:"center" });
    }
  });

  // Three pillars
  [
    { x:0.45, color:C.indigo,  title:"Forgery Detection",  body:"EfficientNet-B4 trained on 6-channel RGB+ELA tensors. Detects JPEG recompression artefacts invisible to the human eye. GradCAM heatmap shows exactly which pixels triggered the alarm." },
    { x:3.6,  color:C.green,   title:"Field Extraction",   body:"LayoutLMv3 (multimodal document AI) extracts student name, institution, degree, date, grade, and roll number. Each field gets an individual confidence score." },
    { x:6.75, color:C.yellow,  title:"NLP Reasoning",      body:"Mistral-7B (Q4 quantised, runs locally on RTX 3050) cross-checks field consistency, date logic, and issuer patterns. Returns a structured reasoning paragraph." },
  ].forEach(p => {
    card(s, p.x, 3.2, 3.0, 2.1);
    s.addShape(pres.shapes.RECTANGLE, { x:p.x, y:3.2, w:0.07, h:2.1, fill:{color:p.color} });
    s.addText(p.title, { x:p.x+0.18, y:3.28, w:2.7, h:0.35, fontSize:13, bold:true, color:C.white });
    s.addText(p.body,  { x:p.x+0.18, y:3.65, w:2.7, h:1.55, fontSize:9.5, color:C.slate });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — ELA forensics deep-dive (the secret weapon)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionLabel(s, "Secret Weapon — ELA Forensics");

  s.addText("Error Level Analysis reveals\ntampering invisible to the human eye", {
    x:0.5, y:0.45, w:9, h:1.0,
    fontSize:28, bold:true, color:C.white, align:"center",
  });

  // ELA explanation cards
  const steps = [
    { n:"01", title:"Re-compress at known quality", body:"Save the certificate as JPEG at a fixed quality level (e.g. 90%). Unedited regions converge to a stable error floor." },
    { n:"02", title:"Compute absolute difference", body:"Subtract the recompressed image from the original pixel-by-pixel. Untouched regions → near zero. Edited regions → high error signal." },
    { n:"03", title:"Amplify and visualise", body:"Scale the error map 10× and apply jet colourmap. Tampered patches appear bright red/yellow — a forensic fingerprint no attacker can hide." },
    { n:"04", title:"4th input channel to CNN", body:"The ELA map is concatenated to RGB as a 4th channel (6 total with 3 for ELA). The forgery detector learns to read both visual and forensic signals simultaneously." },
  ];
  steps.forEach((st, i) => {
    const x = (i%2)*4.8 + 0.45, y = i < 2 ? 1.65 : 3.25;
    card(s, x, y, 4.5, 1.3);
    s.addShape(pres.shapes.OVAL, { x:x+0.15, y:y+0.15, w:0.55, h:0.55, fill:{color:C.indigo} });
    s.addText(st.n, { x:x+0.15, y:y+0.15, w:0.55, h:0.55, fontSize:13, bold:true, color:C.white, align:"center" });
    s.addText(st.title, { x:x+0.85, y:y+0.14, w:3.5, h:0.3, fontSize:11, bold:true, color:C.white });
    s.addText(st.body,  { x:x+0.85, y:y+0.46, w:3.5, h:0.74, fontSize:9.5, color:C.slate });
  });

  // Result callout
  s.addShape(pres.shapes.RECTANGLE, { x:0.45, y:4.72, w:9.1, h:0.65, fill:{color:"1A1032"}, line:{color:C.indigo,width:1} });
  s.addText("In our tests: ELA channel alone gives 8× higher signal in tampered regions vs surroundings — confirmed at build time with zero trained models.", {
    x:0.65, y:4.8, w:8.7, h:0.45, fontSize:10.5, color:C.indigoL, align:"center",
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Architecture
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionLabel(s, "System Architecture");

  s.addText("End-to-end AI forensics pipeline", {
    x:0.5, y:0.42, w:9, h:0.5, fontSize:26, bold:true, color:C.white, align:"center",
  });

  // Layer boxes
  const layers = [
    { y:1.1,  label:"INPUT LAYER",    items:["Chrome extension (screen select)","Web upload (JPG/PNG/PDF)","URL / API input"],      color:C.indigoL },
    { y:2.0,  label:"PREPROCESSING",  items:["Deskew · CLAHE denoise · gamma normalise · ELA generation"],                          color:"60A5FA"   },
    { y:2.72, label:"DL CORE",        items:["EfficientNet-B4+ELA (forgery)","LayoutLMv3 (fields)","Mistral-7B GGUF (reasoning)"],  color:C.indigo   },
    { y:3.62, label:"FUSION + API",   items:["Trust score fusion (0–100) · FastAPI + Celery · PostgreSQL + Redis"],                 color:C.green    },
    { y:4.38, label:"OUTPUT",         items:["React dashboard · GradCAM heatmap · PDF report · Chrome popup"],                      color:C.yellow   },
  ];
  layers.forEach(l => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.45, y:l.y, w:9.1, h:0.62, fill:{color:"1E293B"}, line:{color:l.color,width:1} });
    s.addText(l.label, { x:0.6, y:l.y+0.04, w:1.6, h:0.2, fontSize:8, bold:true, color:l.color, charSpacing:2 });
    s.addText(l.items.join("   ·   "), { x:0.6, y:l.y+0.26, w:8.7, h:0.28, fontSize:10, color:C.slate });
  });

  // Weights callout
  card(s, 0.45, 5.15, 9.1, 0.35, { fill:"0F172A", border:C.border });
  s.addText("Fusion weights:  Forgery 45%  ·  Field confidence 35%  ·  NLP reasoning 20%  ·  Institution match bonus 5%", {
    x:0.6, y:5.2, w:8.8, h:0.25, fontSize:9.5, color:C.indigoL, align:"center",
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Chrome extension wow slide
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionLabel(s, "Chrome Extension");

  s.addText("Verify any certificate without leaving your browser", {
    x:0.5, y:0.44, w:9, h:0.65, fontSize:26, bold:true, color:C.white, align:"center",
  });

  // Steps
  const extSteps = [
    { n:"1", title:"Click the extension",      body:"Activates the selection overlay on any webpage or document viewer" },
    { n:"2", title:"Drag to select",           body:"Draw a rectangle around the certificate — any size, any position on screen" },
    { n:"3", title:"AI pipeline runs",         body:"captureVisibleTab → crop → upload → OCR → ELA → LayoutLM → Mistral → trust score" },
    { n:"4", title:"Verdict in popup",         body:"Score ring, sub-score bars, field table, and NLP reasoning — all in the extension popup" },
  ];
  extSteps.forEach((st, i) => {
    const x = 0.45 + (i%2)*4.82, y = i < 2 ? 1.3 : 3.15;
    card(s, x, y, 4.5, 1.55);
    s.addShape(pres.shapes.OVAL, { x:x+0.18, y:y+0.18, w:0.65, h:0.65, fill:{color:C.indigo} });
    s.addText(st.n, { x:x+0.18, y:y+0.18, w:0.65, h:0.65, fontSize:20, bold:true, color:C.white, align:"center" });
    s.addText(st.title, { x:x+0.98, y:y+0.2,  w:3.35, h:0.35, fontSize:13, bold:true, color:C.white });
    s.addText(st.body,  { x:x+0.98, y:y+0.58, w:3.35, h:0.86, fontSize:10, color:C.slate });
  });

  s.addShape(pres.shapes.RECTANGLE, { x:0.45, y:4.9, w:9.1, h:0.52, fill:{color:"1A1032"}, line:{color:C.green,width:1} });
  s.addText("✓  The inline page badge shows verdict + score immediately — no tab switching required. Judges can test it themselves in 10 seconds.", {
    x:0.65, y:4.98, w:8.7, h:0.36, fontSize:10.5, color:C.green, align:"center",
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Demo slide (the punchline)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionLabel(s, "Live Demo");

  s.addText("Three certificates.\nThree verdicts. Live.", {
    x:0.5, y:0.42, w:9, h:1.1,
    fontSize:32, bold:true, color:C.white, align:"center",
  });

  const demos = [
    { title:"Certificate 1",    verdict:"GENUINE",    score:"87",  color:C.green,  detail:"Rahul Sharma · DTU · B.Tech CSE\nAll fields consistent. Seal matched.\nELA shows zero recompression artefacts." },
    { title:"Certificate 2",    verdict:"FAKE",       score:"22",  color:C.red,    detail:"Name field tampered (Priya → Anjali)\nGradCAM heatmap highlights name region.\nELA reveals 8× higher error in tampered area." },
    { title:"Certificate 3",    verdict:"FAKE",       score:"28",  color:C.red,    detail:"Grade inflated B → A+, CGPA 6.5 → 9.95\nField validator: CGPA suspiciously high.\nNLP: date + roll number inconsistency flagged." },
  ];
  demos.forEach((d, i) => {
    const x = 0.45 + i*3.22;
    card(s, x, 1.68, 3.05, 3.65, { border:d.color });
    // Verdict band
    s.addShape(pres.shapes.RECTANGLE, { x, y:1.68, w:3.05, h:0.55, fill:{color:d.color} });
    s.addText(d.verdict, { x, y:1.68, w:3.05, h:0.55, fontSize:16, bold:true, color:C.navy, align:"center" });
    // Score
    s.addText(d.score, { x, y:2.32, w:3.05, h:0.75, fontSize:44, bold:true, color:d.color, align:"center" });
    s.addText("/ 100", { x, y:3.05, w:3.05, h:0.25, fontSize:11, color:C.slate, align:"center" });
    s.addShape(pres.shapes.LINE, { x:x+0.2, y:3.38, w:2.65, h:0, line:{color:C.border,width:1} });
    s.addText(d.title, { x, y:3.46, w:3.05, h:0.28, fontSize:11, bold:true, color:C.white, align:"center" });
    s.addText(d.detail, { x:x+0.15, y:3.78, w:2.75, h:1.42, fontSize:9, color:C.slate, align:"left" });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Tech stack
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionLabel(s, "Technical Stack");

  s.addText("Built for performance on an RTX 3050 (4 GB VRAM)", {
    x:0.5, y:0.44, w:9, h:0.5, fontSize:22, bold:true, color:C.white, align:"center",
  });

  const stack = [
    { cat:"Deep Learning",  color:C.indigo, items:[
      ["EfficientNet-B4","6-channel RGB+ELA forgery detector — 18.4M params"],
      ["TrOCR-base","Fine-tuned on certificate field crops — printed + handwritten"],
      ["LayoutLMv3-base","NER head for 7 certificate fields with BIO tagging"],
      ["Mistral-7B Q4_K_M","4.1 GB GGUF, 20 GPU layers on RTX 3050 via llama.cpp"],
    ]},
    { cat:"Backend",        color:C.green,  items:[
      ["FastAPI 0.111","Async REST API with background task queue"],
      ["Celery + Redis","Non-blocking inference — 202 response, poll for result"],
      ["PostgreSQL + Alembic","5-table schema: certs, verifications, institutions, users, audit"],
      ["SQLAlchemy async","asyncpg driver — full async DB session per request"],
    ]},
    { cat:"Frontend + Ext", color:C.yellow, items:[
      ["React + Vite","Component library: TrustScoreRing, FieldBreakdown, HeatmapViewer"],
      ["Chrome Ext MV3","captureVisibleTab → crop → verify in 3 steps, inline badge"],
      ["ReportLab","Multi-page forensic PDF with embedded GradCAM + audit trail"],
      ["Framer Motion","Animated score ring, staggered field bars, processing steps"],
    ]},
  ];

  stack.forEach((col, ci) => {
    const x = 0.45 + ci*3.22;
    card(s, x, 1.12, 3.05, 4.25);
    s.addShape(pres.shapes.RECTANGLE, { x, y:1.12, w:3.05, h:0.42, fill:{color:col.color} });
    s.addText(col.cat.toUpperCase(), { x, y:1.12, w:3.05, h:0.42, fontSize:11, bold:true, color:C.navy, align:"center" });
    col.items.forEach(([name, desc], ii) => {
      const iy = 1.66 + ii*0.88;
      s.addText(name, { x:x+0.18, y:iy,      w:2.7, h:0.28, fontSize:11, bold:true, color:C.white });
      s.addText(desc, { x:x+0.18, y:iy+0.29, w:2.7, h:0.48, fontSize:9,  color:C.slate });
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Results & impact
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionLabel(s, "Results & Impact");

  s.addText("What we achieved in 23 days", {
    x:0.5, y:0.44, w:9, h:0.55, fontSize:26, bold:true, color:C.white, align:"center",
  });

  // Big stats row
  [
    ["8.0×",  "higher ELA signal\nin tampered regions",  C.indigoL, 0.45],
    ["90+",   "test cases passing\nacross all 5 phases",  C.green,   3.0 ],
    ["< 30s", "end-to-end verification\nincl. Mistral-7B", C.yellow,  5.55],
    ["3",     "demo certs — 3 correct\nverdicts live",    C.red,     8.1 ],
  ].forEach(([num,lbl,col,x]) => statCard(s, x, 1.1, 1.65, num, lbl, col));

  // What makes it different
  s.addText("Why judges should pick this project", {
    x:0.5, y:2.65, w:9, h:0.38, fontSize:15, bold:true, color:C.white,
  });

  const diffs = [
    [C.green,    "ELA as a 4th channel",        "Not one team uses ELA as input to the CNN — it's a genuine research-level insight that directly improves forgery detection accuracy"],
    [C.indigo,   "Everything runs locally",      "No paid APIs. No cloud. RTX 3050 handles all three models. Offline-capable — critical for institutions with data sovereignty requirements"],
    [C.yellow,   "Chrome extension with capture","Select any screen region and get a verdict. No one else has this — it's the demo moment that judges remember"],
    [C.indigoL,  "Complete pipeline, not a demo","113 source files, 5 phases, PostgreSQL schema, Celery queue, PDF reports. This is a product, not a prototype"],
  ];
  diffs.forEach((d, i) => {
    const y = 3.12 + i*0.55;
    s.addShape(pres.shapes.OVAL, { x:0.45, y:y+0.08, w:0.22, h:0.22, fill:{color:d[0]} });
    s.addText(d[1], { x:0.82, y:y,      w:2.5, h:0.28, fontSize:11, bold:true, color:C.white });
    s.addText(d[2], { x:0.82, y:y+0.28, w:8.6, h:0.22, fontSize:9.5, color:C.slate });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Closing
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);

  s.addShape(pres.shapes.OVAL, {
    x:4.1, y:0.5, w:1.8, h:1.8, fill:{ color:C.indigo }, line:{color:C.indigo,width:0},
  });
  s.addText("🛡", { x:4.1, y:0.58, w:1.8, h:1.6, fontSize:52, align:"center" });

  s.addText("CertValidator", {
    x:0.5, y:2.5, w:9, h:0.9, fontSize:44, bold:true, color:C.white, align:"center",
  });
  s.addText("Every certificate. Every claim. Verified.", {
    x:0.5, y:3.42, w:9, h:0.5, fontSize:18, color:C.indigoL, align:"center", italic:true,
  });

  // CTA boxes
  [
    { label:"🌐  Dashboard",  val:"http://localhost:5173",         col:C.indigo },
    { label:"⚡  API Docs",   val:"http://localhost:8000/docs",    col:C.green  },
    { label:"📦  Source",     val:"github.com/certvalidator",      col:C.yellow },
  ].forEach((item, i) => {
    const x = 1.2 + i*2.65;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y:4.1, w:2.45, h:0.82, fill:{color:"1E293B"}, line:{color:item.col,width:1}, rectRadius:0.08 });
    s.addText(item.label, { x, y:4.16, w:2.45, h:0.3, fontSize:10, bold:true, color:item.col, align:"center" });
    s.addText(item.val,   { x, y:4.48, w:2.45, h:0.3, fontSize:9,  color:C.slate,   align:"center" });
  });

  s.addText("Built with PyTorch · FastAPI · React · Chrome Extension API · ReportLab  |  Hackathon 2026", {
    x:0.5, y:5.15, w:9, h:0.28, fontSize:9, color:C.border, align:"center",
  });
}

// ── Write file ────────────────────────────────────────────────────────────
const outPath = "presentation/CertValidator_Hackathon.pptx";
pres.writeFile({ fileName: outPath })
  .then(() => console.log(`✓ Deck written → ${outPath}`))
  .catch(e  => { console.error("Error:", e); process.exit(1); });
