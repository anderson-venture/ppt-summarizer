import { Marked } from "marked";
import fs from "node:fs";
import path from "node:path";
import { CONFIG } from "./config.js";

const marked = new Marked();

// Render mermaid code blocks as <div class="mermaid"> for client-side rendering
marked.use({
  renderer: {
    code({ text, lang, escaped }) {
      if (lang === "mermaid") {
        return `<div class="mermaid">\n${text}\n</div>\n`;
      }
      const langString = lang ?? "";
      const code = escaped ? text : escapeHtml(text);
      if (!langString) {
        return `<pre><code>${code}</code></pre>\n`;
      }
      return `<pre><code class="language-${escapeHtml(langString)}">${code}</code></pre>\n`;
    },
  },
});

function escapeHtml(html: string): string {
  const map: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  return html.replace(/[&<>"']/g, (ch) => map[ch] ?? ch);
}

export type OutputFormat = "pdf" | "html";

export async function generateOutput(
  markdown: string,
  outputPath: string,
  format: OutputFormat
): Promise<void> {
  const html = await markdownToHtml(markdown);
  const styledHtml = wrapInTemplate(html);

  if (format === "html") {
    fs.writeFileSync(outputPath, styledHtml);
    console.log(`[out] HTML saved to ${outputPath}`);
    return;
  }

  // PDF path — lazy-import puppeteer so HTML mode never needs Chrome
  console.log("[out] Rendering PDF via Puppeteer...");
  const puppeteer = await import("puppeteer");
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  try {
    const page = await browser.newPage();
    await page.setContent(styledHtml, { waitUntil: "networkidle0" });

    // Give Mermaid time to render diagrams (client-side JS)
    if (styledHtml.includes('class="mermaid"')) {
      await page
        .waitForFunction(
          () => {
            const el = document.querySelector(".mermaid");
            return !el || el.querySelector("svg");
          },
          { timeout: 10000 }
        )
        .catch(() => {
          /* continue if timeout - diagrams may still render */
        });
    }

    await page.pdf({
      path: outputPath,
      format: "A4",
      margin: { top: "20mm", bottom: "20mm", left: "18mm", right: "18mm" },
      printBackground: true,
      displayHeaderFooter: true,
      headerTemplate: `<span></span>`,
      footerTemplate: `
        <div style="font-size:9px; color:#888; width:100%; text-align:center; padding:0 20mm;">
          <span class="pageNumber"></span> / <span class="totalPages"></span>
        </div>`,
    });

    console.log(`[out] PDF saved to ${outputPath}`);
  } finally {
    await browser.close();
  }
}

async function markdownToHtml(md: string): Promise<string> {
  const imgPattern = /!\[([^\]]*)\]\(([^)]+)\)/g;
  let processed = md;

  for (const match of md.matchAll(imgPattern)) {
    const [full, alt, src] = match;
    const imgPath = path.join(CONFIG.imagesDir, src);

    if (fs.existsSync(imgPath)) {
      const data = fs.readFileSync(imgPath);
      const b64 = data.toString("base64");
      const ext = path.extname(src).slice(1) || "png";
      const dataUri = `data:image/${ext};base64,${b64}`;
      const replacement = `<figure><img src="${dataUri}" alt="${alt}"></figure>`;
      processed = processed.replace(full, replacement);
    }
  }

  return await marked.parse(processed);
}

function wrapInTemplate(bodyHtml: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Study Notes</title>
<style>
  @page {
    size: A4;
    margin: 0;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #2d3748;
    background: #ffffff;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 32px;
  }

  @media print {
    body { 
      max-width: 100%; 
      padding: 20px;
    }
  }

  @media (max-width: 768px) {
    body {
      padding: 24px 20px;
    }
  }

  /* Typography */
  h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #1a202c;
    border-bottom: 3px solid #4299e1;
    padding-bottom: 12px;
    margin-top: 32px;
    margin-bottom: 20px;
    page-break-after: avoid;
  }

  h1:first-child {
    margin-top: 0;
  }

  h2 {
    font-size: 20pt;
    font-weight: 600;
    color: #2c5282;
    margin-top: 32px;
    margin-bottom: 16px;
    padding-left: 12px;
    border-left: 4px solid #4299e1;
    page-break-after: avoid;
  }

  h3 {
    font-size: 15pt;
    font-weight: 600;
    color: #4a5568;
    margin-top: 24px;
    margin-bottom: 12px;
    page-break-after: avoid;
  }

  h4 {
    font-size: 13pt;
    font-weight: 600;
    color: #718096;
    margin-top: 20px;
    margin-bottom: 10px;
  }

  p {
    margin: 12px 0;
    text-align: justify;
  }

  /* Lists */
  ul, ol {
    margin: 12px 0;
    padding-left: 28px;
  }

  li {
    margin-bottom: 8px;
    line-height: 1.6;
  }

  ul li {
    list-style-type: disc;
  }

  ol li {
    list-style-type: decimal;
  }

  /* Contents section styling */
  #contents {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* Section headers in Contents */
  #contents ~ * h2,
  h1 + h2 {
    background: #f7fafc;
    padding: 12px 16px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin-left: -12px;
  }

  /* Links */
  a {
    color: #4299e1;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.2s;
  }

  a:hover {
    border-bottom-color: #4299e1;
  }

  /* Text formatting */
  strong {
    color: #1a202c;
    font-weight: 600;
  }

  em {
    color: #4a5568;
    font-style: italic;
  }

  /* Code */
  code {
    background: #edf2f7;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10pt;
    font-family: 'Courier New', monospace;
    color: #e53e3e;
  }

  pre {
    background: #1a202c;
    color: #e2e8f0;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 16px;
    overflow-x: auto;
    font-size: 10pt;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }

  pre code {
    background: transparent;
    color: inherit;
    padding: 0;
  }

  /* Images */
  figure {
    margin: 24px 0;
    text-align: center;
    page-break-inside: avoid;
    background: #f7fafc;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
  }

  figure img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }

  figcaption {
    font-size: 10pt;
    color: #718096;
    margin-top: 8px;
    font-style: italic;
  }

  /* Tables */
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 10pt;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border-radius: 8px;
    overflow: hidden;
  }

  th, td {
    border: 1px solid #e2e8f0;
    padding: 12px 16px;
    text-align: left;
  }

  th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
  }

  tr:nth-child(even) {
    background: #f7fafc;
  }

  tr:hover {
    background: #edf2f7;
  }

  /* Horizontal rule */
  hr {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 32px 0;
  }

  /* Blockquote */
  blockquote {
    border-left: 4px solid #4299e1;
    margin: 20px 0;
    padding: 12px 20px;
    background: #ebf8ff;
    color: #2c5282;
    border-radius: 4px;
    font-style: italic;
  }

  /* Mermaid diagrams */
  .mermaid {
    margin: 32px 0;
    text-align: center;
    background: #f7fafc;
    padding: 24px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    page-break-inside: avoid;
  }

  .mermaid svg {
    max-width: 100%;
    height: auto;
  }

  /* Review Questions, Glossary, Common Pitfalls sections */
  h1[id*="review"],
  h1[id*="glossary"],
  h1[id*="pitfalls"] {
    background: #f7fafc;
    padding: 16px 20px;
    border-radius: 8px;
    margin-top: 40px;
  }

  /* Special styling for Q&A - using class-based approach */
  .question {
    margin: 16px 0;
    padding: 12px 16px;
    border-radius: 6px;
    background: #ebf8ff;
    border-left: 4px solid #4299e1;
  }

  .answer {
    margin: 16px 0;
    padding: 12px 16px;
    border-radius: 6px;
    background: #f0fff4;
    border-left: 4px solid #48bb78;
  }

  /* Print optimizations */
  @media print {
    h1, h2, h3 {
      page-break-after: avoid;
    }
    
    figure, .mermaid {
      page-break-inside: avoid;
    }
    
    body {
      font-size: 10pt;
    }
  }
</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
${bodyHtml}
<script>
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ 
      startOnLoad: true, 
      theme: 'default',
      themeVariables: {
        primaryColor: '#4299e1',
        primaryTextColor: '#1a202c',
        primaryBorderColor: '#2c5282',
        lineColor: '#4299e1',
        secondaryColor: '#ebf8ff',
        tertiaryColor: '#f7fafc'
      },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      }
    });
  }
</script>
</body>
</html>`;
}
