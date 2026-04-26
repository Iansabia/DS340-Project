#!/bin/bash
# Build a polished research-paper PDF from PAPER_DRAFT.md.
# Pipeline: markdown → standalone HTML (with academic CSS) → PDF (Chrome headless).
#
# Output: paper.pdf at the repo root.
#
# Why this path: pandoc → LaTeX choked on unescaped & in author names
# (Wolfers & Zitzewitz, etc.) and required setspace.sty + a maintained
# TeX Live install. HTML+Chrome is more portable on macOS.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_HTML=/tmp/paper_build.html
OUT_PDF=paper.pdf
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

cat > /tmp/paper_style.css <<'EOF'
@page {
  size: letter;
  margin: 1in 1.1in 1in 1.1in;
  @top-left { content: "Complexity Is Not an Edge"; font-size: 9pt; color: #666; }
  @top-right { content: "Sabia & Jang · DS340 Spring 2026"; font-size: 9pt; color: #666; }
  @bottom-center { content: counter(page); font-size: 10pt; color: #555; }
}

@page :first {
  @top-left { content: none; }
  @top-right { content: none; }
}

html, body {
  font-family: "Charter", "Iowan Old Style", "Source Serif Pro", "STIX Two Text", Georgia, serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #1a1a1a;
  background: white;
}

body {
  max-width: none;
  margin: 0;
  padding: 0;
}

h1, h2, h3, h4, h5, h6 {
  font-family: "Inter", "Helvetica Neue", "Arial", sans-serif;
  color: #0a0a0a;
  page-break-after: avoid;
}

h1 {
  font-size: 22pt;
  font-weight: 700;
  margin-top: 0;
  margin-bottom: 0.3em;
  line-height: 1.2;
}

/* Title block */
h1:first-of-type {
  text-align: left;
  border-bottom: 2px solid #0a0a0a;
  padding-bottom: 0.4em;
  margin-bottom: 0.5em;
}

h2 {
  font-size: 14pt;
  font-weight: 600;
  margin-top: 1.6em;
  margin-bottom: 0.4em;
  border-bottom: 1px solid #d0d0d0;
  padding-bottom: 0.15em;
}

h3 {
  font-size: 12pt;
  font-weight: 600;
  margin-top: 1.2em;
  margin-bottom: 0.3em;
}

h4 {
  font-size: 11pt;
  font-weight: 600;
  margin-top: 0.9em;
  margin-bottom: 0.2em;
}

p { margin: 0 0 0.6em 0; text-align: justify; orphans: 3; widows: 3; hyphens: auto; }

strong { font-weight: 600; }
em { font-style: italic; }

/* Code */
code {
  font-family: "JetBrains Mono", "Source Code Pro", "Menlo", monospace;
  font-size: 9.5pt;
  background: #f4f4f4;
  padding: 1px 4px;
  border-radius: 2px;
  word-break: break-word;
}

pre {
  background: #f7f7f7;
  border: 1px solid #e0e0e0;
  border-radius: 3px;
  padding: 8px 12px;
  font-size: 9pt;
  line-height: 1.4;
  overflow-x: auto;
  page-break-inside: avoid;
}

pre code {
  background: transparent;
  padding: 0;
  border-radius: 0;
}

/* Tables */
table {
  border-collapse: collapse;
  margin: 0.8em auto;
  font-size: 9.5pt;
  page-break-inside: avoid;
  width: 100%;
}

th, td {
  border: 1px solid #c0c0c0;
  padding: 4px 8px;
  text-align: left;
  vertical-align: top;
}

th {
  background: #efefef;
  font-weight: 600;
  font-family: "Inter", sans-serif;
}

tr:nth-child(even) td { background: #fafafa; }

/* Lists */
ul, ol { margin: 0.4em 0 0.6em 1.5em; padding: 0; }
li { margin-bottom: 0.2em; }

/* Blockquote */
blockquote {
  border-left: 3px solid #c0c0c0;
  margin: 0.8em 0;
  padding: 0.2em 0 0.2em 1em;
  color: #444;
  font-style: italic;
}

/* Figures */
img {
  display: block;
  margin: 0.8em auto;
  max-width: 100%;
  page-break-inside: avoid;
}

/* Horizontal rules */
hr {
  border: 0;
  border-top: 1px solid #d0d0d0;
  margin: 1.5em 0;
}

/* Links */
a { color: #1a4d8c; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Math */
.math.display {
  text-align: center;
  margin: 0.6em 0;
}

/* Footnotes */
.footnote-ref { font-size: 0.8em; vertical-align: super; }
.footnotes {
  font-size: 9.5pt;
  margin-top: 2em;
  border-top: 1px solid #c0c0c0;
  padding-top: 0.6em;
}

/* Avoid orphan headings */
h2 + p, h3 + p, h4 + p { page-break-before: avoid; }
EOF

echo "[1/3] Rendering markdown → HTML..."
pandoc PAPER_DRAFT.md \
  -o "$OUT_HTML" \
  --standalone \
  --self-contained \
  --css=/tmp/paper_style.css \
  --metadata title="Complexity Is Not an Edge" \
  --section-divs \
  --mathjax \
  -f markdown+pipe_tables+raw_html

echo "[2/3] Rendering HTML → PDF (Chrome headless)..."
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --hide-scrollbars \
  --virtual-time-budget=5000 \
  --print-to-pdf="$OUT_PDF" \
  "file://$(pwd)/${OUT_HTML#/tmp/}" 2>&1 | tail -2 || true

# Chrome has trouble with /tmp paths sometimes; copy to repo dir
cp "$OUT_HTML" /tmp/paper_build_local.html
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --hide-scrollbars \
  --virtual-time-budget=5000 \
  --print-to-pdf="$OUT_PDF" \
  "file:///tmp/paper_build_local.html" 2>&1 | tail -3

echo "[3/3] Verify..."
if [ -f "$OUT_PDF" ]; then
  PAGES=$(pdfinfo "$OUT_PDF" 2>/dev/null | awk '/^Pages:/ {print $2}')
  SIZE=$(/usr/bin/du -h "$OUT_PDF" | awk '{print $1}')
  echo "✓ Wrote $OUT_PDF ($SIZE, $PAGES pages)"
else
  echo "✗ Build failed"
  exit 1
fi
