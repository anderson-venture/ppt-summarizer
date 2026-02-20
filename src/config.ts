import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

export const CONFIG = {
  inputDir: path.join(ROOT, "input"),
  outputDir: path.join(ROOT, "output"),
  imagesDir: path.join(ROOT, "output", "images"),

  model: "gpt-4o-mini" as const,
  visionDetail: "low" as const,
  visionBatchSize: 10,
  maxConcurrentBatches: 4,

  imageMaxWidth: 768,
} as const;
