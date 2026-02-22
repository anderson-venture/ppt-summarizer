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
      const replacement = `<figure><img src="${dataUri}" alt="${alt}"><figcaption>${alt}</figcaption></figure>`;
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
<title>Study Notes</title>
<style>
  @page {
    size: A4;
    margin: 0;
  }

  * {
    box-sizing: border-box;
  }

  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 860px;
    margin: 0 auto;
    padding: 24px;
  }

  @media print {
    body { max-width: 100%; padding: 0; }
  }

  h1 {
    font-size: 20pt;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 6px;
    margin-top: 28px;
    margin-bottom: 12px;
    color: #1e3a5f;
    page-break-after: avoid;
  }

  h2 {
    font-size: 15pt;
    color: #1e40af;
    margin-top: 22px;
    margin-bottom: 8px;
    border-bottom: 1px solid #dbeafe;
    padding-bottom: 4px;
    page-break-after: avoid;
  }

  h3 {
    font-size: 12.5pt;
    color: #374151;
    margin-top: 16px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }

  p {
    margin: 6px 0;
  }

  ul, ol {
    margin: 4px 0 8px 0;
    padding-left: 22px;
  }

  li {
    margin-bottom: 3px;
  }

  strong {
    color: #111827;
  }

  code {
    background: #f3f4f6;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10pt;
  }

  pre {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 10px;
    overflow-x: auto;
    font-size: 9.5pt;
  }

  figure {
    margin: 14px 0;
    text-align: center;
    page-break-inside: avoid;
  }

  figure img {
    max-width: 92%;
    height: auto;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
  }

  figcaption {
    font-size: 9pt;
    color: #6b7280;
    margin-top: 4px;
    font-style: italic;
  }

  table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 10pt;
  }

  th, td {
    border: 1px solid #d1d5db;
    padding: 6px 10px;
    text-align: left;
  }

  th {
    background: #f3f4f6;
    font-weight: 600;
  }

  hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 18px 0;
  }

  blockquote {
    border-left: 3px solid #2563eb;
    margin: 10px 0;
    padding: 4px 12px;
    background: #eff6ff;
    color: #1e40af;
  }

  .mermaid {
    margin: 16px 0;
    text-align: center;
  }

  .mermaid svg {
    max-width: 100%;
    height: auto;
  }
</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
${bodyHtml}
<script>
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
  }
</script>
</body>
</html>`;
}
