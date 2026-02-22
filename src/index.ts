import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { CONFIG } from "./config.js";
import { processPDF } from "./pdf-processor.js";
import {
  analyzeAndSynthesize,
  type SynthesisPageInput,
} from "./llm-analyzer.js";
import { generateOutput, type OutputFormat } from "./pdf-writer.js";

async function main() {
  const t0 = performance.now();

  const { inputPath, format } = parseArgs();
  const pdfPath = resolveInput(inputPath);
  console.log(`\nInput:  ${pdfPath}`);
  console.log(`Format: ${format}\n`);

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey || apiKey === "your-api-key-here") {
    console.error("Error: Set OPENAI_API_KEY in .env");
    process.exit(1);
  }

  fs.mkdirSync(CONFIG.outputDir, { recursive: true });

  // ── Step 1: Extract text + images from PDF ──
  const t1 = performance.now();
  const { pages, totalImages } = await processPDF(pdfPath);
  const step1Time = ((performance.now() - t1) / 1000).toFixed(1);
  console.log(`  Step 1 took ${step1Time}s\n`);

  // ── Step 2 & 3: LLM vision analysis + note synthesis ──
  const {
    studyNotes,
    costEstimate,
    imageDescriptions,
    synthesisInputPages,
  } = await analyzeAndSynthesize(pages, apiKey);

  // Save intermediate results
  const imageDescPath = path.join(CONFIG.outputDir, "image-descriptions.json");
  fs.writeFileSync(
    imageDescPath,
    JSON.stringify(imageDescriptions, null, 2),
    "utf-8"
  );
  console.log(`[out] Image descriptions saved to ${imageDescPath}`);

  const synthInputPath = path.join(
    CONFIG.outputDir,
    "synthesis-input-pages.json"
  );
  fs.writeFileSync(
    synthInputPath,
    JSON.stringify(synthesisInputPages, null, 2),
    "utf-8"
  );
  console.log(`[out] Synthesis input (per-page) saved to ${synthInputPath}`);

  // Also save a readable markdown version of synthesis input
  const synthInputMdPath = path.join(
    CONFIG.outputDir,
    "synthesis-input-pages.md"
  );
  fs.writeFileSync(
    synthInputMdPath,
    formatSynthesisInputMarkdown(synthesisInputPages),
    "utf-8"
  );
  console.log(`[out] Synthesis input (readable) saved to ${synthInputMdPath}`);

  // Always save the raw markdown
  const mdPath = path.join(CONFIG.outputDir, "study-notes.md");
  fs.writeFileSync(mdPath, studyNotes);
  console.log(`[out] Markdown saved to ${mdPath}`);

  // ── Step 4: Generate final output ──
  const t3 = performance.now();
  const ext = format === "html" ? "html" : "pdf";
  const outFile = path.join(CONFIG.outputDir, `study-notes.${ext}`);
  await generateOutput(studyNotes, outFile, format);
  const step4Time = ((performance.now() - t3) / 1000).toFixed(1);
  console.log(`  Step 4 took ${step4Time}s\n`);

  // ── Summary ──
  const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
  console.log("─".repeat(48));
  console.log(`  Total time:   ${totalTime}s`);
  console.log(`  Pages:        ${pages.length}`);
  console.log(`  Images:       ${totalImages}`);
  console.log(`  API cost:     $${costEstimate.toFixed(4)}`);
  console.log(`  Output:       ${outFile}`);
  console.log("─".repeat(48));
}

function formatSynthesisInputMarkdown(pages: SynthesisPageInput[]): string {
  const parts: string[] = [];
  for (const p of pages) {
    parts.push(`## Page ${p.pageNumber}\n`);
    if (p.text) parts.push(p.text.trim(), "\n");
    for (const img of p.images) {
      parts.push(`### ${img.filename}\n`, img.description, "\n");
    }
    parts.push("\n");
  }
  return parts.join("");
}

function parseArgs(): { inputPath?: string; format: OutputFormat } {
  const args = process.argv.slice(2);
  let inputPath: string | undefined;
  let format: OutputFormat = "pdf";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--format" && args[i + 1]) {
      const val = args[i + 1].toLowerCase();
      if (val !== "pdf" && val !== "html") {
        console.error(`Invalid format "${val}". Use "pdf" or "html".`);
        process.exit(1);
      }
      format = val;
      i++;
    } else if (!args[i].startsWith("--")) {
      inputPath = args[i];
    }
  }

  return { inputPath, format };
}

function resolveInput(arg?: string): string {
  if (arg) {
    const abs = path.resolve(arg);
    if (!fs.existsSync(abs)) {
      console.error(`File not found: ${abs}`);
      process.exit(1);
    }
    return abs;
  }

  const files = fs
    .readdirSync(CONFIG.inputDir)
    .filter((f) => f.toLowerCase().endsWith(".pdf"));

  if (files.length === 0) {
    console.error(
      `No PDF found. Either:\n` +
        `  - Place a PDF in ./input/\n` +
        `  - Or pass a path: npx tsx src/index.ts path/to/file.pdf`
    );
    process.exit(1);
  }

  if (files.length > 1) {
    console.log(`Multiple PDFs found, using first: ${files[0]}`);
  }

  return path.join(CONFIG.inputDir, files[0]);
}

main().catch((err) => {
  console.error("\nFatal error:", err);
  process.exit(1);
});
