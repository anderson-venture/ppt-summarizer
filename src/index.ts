import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { CONFIG } from "./config.js";
import { processPDF } from "./pdf-processor.js";
import { analyzeAndSynthesize } from "./llm-analyzer.js";
import { generatePDF } from "./pdf-writer.js";

async function main() {
  const t0 = performance.now();

  // ── Resolve input PDF ──
  const inputArg = process.argv[2];
  const pdfPath = resolveInput(inputArg);
  console.log(`\nInput: ${pdfPath}\n`);

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
  const t2 = performance.now();
  const { studyNotes, costEstimate } = await analyzeAndSynthesize(
    pages,
    apiKey
  );
  const step23Time = ((performance.now() - t2) / 1000).toFixed(1);
  console.log(`  Steps 2-3 took ${step23Time}s\n`);

  // Save the raw markdown as well
  const mdPath = path.join(CONFIG.outputDir, "study-notes.md");
  fs.writeFileSync(mdPath, studyNotes);
  console.log(`[out] Markdown saved to ${mdPath}`);

  // ── Step 4: Generate PDF ──
  const t3 = performance.now();
  const outPdf = path.join(CONFIG.outputDir, "study-notes.pdf");
  await generatePDF(studyNotes, outPdf);
  const step4Time = ((performance.now() - t3) / 1000).toFixed(1);
  console.log(`  Step 4 took ${step4Time}s\n`);

  // ── Summary ──
  const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
  console.log("─".repeat(48));
  console.log(`  Total time:   ${totalTime}s`);
  console.log(`  Pages:        ${pages.length}`);
  console.log(`  Images:       ${totalImages}`);
  console.log(`  API cost:     $${costEstimate.toFixed(4)}`);
  console.log(`  Output:       ${outPdf}`);
  console.log("─".repeat(48));
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

  // Look for any PDF in the input directory
  const files = fs.readdirSync(CONFIG.inputDir).filter((f) =>
    f.toLowerCase().endsWith(".pdf")
  );

  if (files.length === 0) {
    console.error(
      `No PDF found. Either:\n` +
        `  - Place a PDF in ./input/\n` +
        `  - Or pass a path: npm run generate -- path/to/file.pdf`
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
