import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

export const CONFIG = {
  inputDir: path.join(ROOT, "input"),
  outputDir: path.join(ROOT, "output"),
  imagesDir: path.join(ROOT, "output", "images"),

  model: "gpt-4o-mini" as const,
  premiumModel: "gpt-4o" as const,
  visionDetail: "low" as const,
  visionBatchSize: 5,
  
  // Images exceeding either threshold are routed to the premium model
  complexImageMinWidth: 500,
  complexImageMinBytes: 102_400, // 100 KB

  imageMaxWidth: 512,
  imageJpegQuality: 82,
} as const;
