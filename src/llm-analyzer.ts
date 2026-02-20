import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { CONFIG } from "./config.js";
import type { PageData, ExtractedImage } from "./pdf-processor.js";

const VISION_TOKENS_PER_LOW_IMAGE = 2833;
const INPUT_COST_PER_TOKEN = 0.15 / 1_000_000;
const OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000;

export interface ImageDescription {
  imageId: string;
  pageNumber: number;
  description: string;
}

export interface AnalysisResult {
  imageDescriptions: ImageDescription[];
  studyNotes: string;
  costEstimate: number;
}

export async function analyzeAndSynthesize(
  pages: PageData[],
  apiKey: string
): Promise<AnalysisResult> {
  const client = new OpenAI({ apiKey });
  let totalCost = 0;

  // ── Step 1: Describe images via vision ──
  const allImages = pages.flatMap((p) =>
    p.images.map((img) => ({
      ...img,
      pageText: p.text,
    }))
  );

  let imageDescriptions: ImageDescription[] = [];

  if (allImages.length > 0) {
    console.log(
      `[llm] Analyzing ${allImages.length} images in batches of ${CONFIG.visionBatchSize}...`
    );

    const batches = chunk(allImages, CONFIG.visionBatchSize);
    const batchPromises = batches.map((batch, idx) =>
      describeImageBatch(client, batch, idx, batches.length)
    );

    const batchResults = await Promise.all(batchPromises);
    for (const result of batchResults) {
      imageDescriptions.push(...result.descriptions);
      totalCost += result.cost;
    }
  } else {
    console.log("[llm] No images found, skipping vision analysis.");
  }

  // ── Step 2: Synthesize study notes ──
  console.log("[llm] Synthesizing study notes...");
  const { notes, cost: synthCost } = await synthesizeNotes(
    client,
    pages,
    imageDescriptions
  );
  totalCost += synthCost;

  console.log(`[llm] Total estimated API cost: $${totalCost.toFixed(4)}`);

  return { imageDescriptions, studyNotes: notes, costEstimate: totalCost };
}

// ── Vision batch processing ──

interface ImageWithContext extends ExtractedImage {
  pageText: string;
}

interface BatchResult {
  descriptions: ImageDescription[];
  cost: number;
}

async function describeImageBatch(
  client: OpenAI,
  batch: ImageWithContext[],
  batchIdx: number,
  totalBatches: number
): Promise<BatchResult> {
  const content: OpenAI.Chat.Completions.ChatCompletionContentPart[] = [];

  content.push({
    type: "text",
    text: buildVisionPrompt(batch),
  });

  for (const img of batch) {
    const base64 = img.buffer.toString("base64");
    content.push({
      type: "image_url",
      image_url: {
        url: `data:image/png;base64,${base64}`,
        detail: CONFIG.visionDetail,
      },
    });
  }

  const messages: ChatCompletionMessageParam[] = [
    {
      role: "user",
      content,
    },
  ];

  const resp = await client.chat.completions.create({
    model: CONFIG.model,
    messages,
    max_tokens: 3000,
    temperature: 0.2,
  });

  const text = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const cost =
    inputTokens * INPUT_COST_PER_TOKEN + outputTokens * OUTPUT_COST_PER_TOKEN;

  console.log(
    `  Batch ${batchIdx + 1}/${totalBatches}: ${inputTokens} in / ${outputTokens} out ($${cost.toFixed(4)})`
  );

  const descriptions = parseImageDescriptions(text, batch);
  return { descriptions, cost };
}

function buildVisionPrompt(batch: ImageWithContext[]): string {
  const imageList = batch
    .map((img, i) => {
      const ctx = img.pageText
        ? `\n   Page text: "${img.pageText.slice(0, 200)}"`
        : "";
      return `${i + 1}. Image ID: ${img.id} (page ${img.pageNumber})${ctx}`;
    })
    .join("\n");

  return `You are analyzing images extracted from university lecture slides. These are scientific diagrams essential for exam study.

For each image, write a detailed description (3-5 sentences) covering:
- The specific biological/scientific concept the diagram illustrates
- ALL labels, terms, arrows, and data visible in the image
- The process or relationship being shown (e.g., steps, pathways, cycles)
- Any key facts a student should memorize from this figure

Images to analyze:
${imageList}

The images follow in the same order. Respond in this exact format:

[${batch[0].id}]
<detailed description>

[${batch[1]?.id ?? "..."}]
<detailed description>

Continue for every image.`;
}

function parseImageDescriptions(
  text: string,
  batch: ImageWithContext[]
): ImageDescription[] {
  const descriptions: ImageDescription[] = [];

  for (const img of batch) {
    const marker = `[${img.id}]`;
    const idx = text.indexOf(marker);
    if (idx === -1) {
      // Fallback: assign whatever text we have for single-image batches
      if (batch.length === 1) {
        descriptions.push({
          imageId: img.id,
          pageNumber: img.pageNumber,
          description: text.trim(),
        });
      }
      continue;
    }

    const start = idx + marker.length;
    const nextMarkerPattern = /\n\[img-/;
    const nextMatch = text.slice(start).search(nextMarkerPattern);
    const end = nextMatch === -1 ? text.length : start + nextMatch;
    const desc = text.slice(start, end).trim();

    descriptions.push({
      imageId: img.id,
      pageNumber: img.pageNumber,
      description: desc,
    });
  }

  return descriptions;
}

// ── Study note synthesis ──

async function synthesizeNotes(
  client: OpenAI,
  pages: PageData[],
  imageDescriptions: ImageDescription[]
): Promise<{ notes: string; cost: number }> {
  const descMap = new Map<string, ImageDescription>();
  for (const d of imageDescriptions) {
    descMap.set(d.imageId, d);
  }

  // Build compact input: only pages with content
  const textContent = pages
    .filter((p) => p.text.length > 0 || p.images.length > 0)
    .map((p) => {
      let entry = `[p${p.pageNumber}]`;
      if (p.text) entry += ` ${p.text}`;
      for (const img of p.images) entry += ` {{${img.filename}}}`;
      return entry;
    })
    .join("\n");

  const allImages = pages.flatMap((p) => p.images);
  const imageCatalog = allImages
    .map((img) => {
      const desc = descMap.get(img.id);
      return `- ${img.filename}: ${desc?.description ?? "diagram/figure"}`;
    })
    .join("\n");

  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: `You are an expert at condensing university lectures into short, high-yield exam study notes. Distill slides into a compact cheat-sheet a student can review in 15 minutes. Diagrams are placed INLINE next to the concept they illustrate — never in a separate section.`,
    },
    {
      role: "user",
      content: `Summarize this ${pages.length}-slide lecture into a concise study note.

CRITICAL FORMATTING RULE:
- Each diagram MUST be placed IMMEDIATELY AFTER the bullet point or paragraph that discusses that concept.
- For example, when you explain transformation, place the transformation diagram right there — NOT in a separate "diagrams" section.
- NEVER create a dedicated images/diagrams section. Images must be woven into the text.

OTHER RULES:
1. Condense aggressively. Merge related slides. Keep only exam-relevant material.
2. Use markdown: # title, ## major topics, ### subtopics. Bullet points for facts. Tables for comparisons.
3. Include key definitions, classifications, numerical facts (pH, temperatures, gene counts, mutation rates), and step-by-step processes.
4. Pick ~8-10 most important diagrams from the IMAGE CATALOG. Embed as ![caption](FILENAME) inline right after the relevant text.
5. Use **bold** for key terms. Keep explanations to 1-2 sentences max.

--- SLIDE TEXT ---
${textContent}

--- IMAGE CATALOG (${allImages.length} images available) ---
${imageCatalog}`,
    },
  ];

  const resp = await client.chat.completions.create({
    model: CONFIG.model,
    messages,
    max_tokens: 4000,
    temperature: 0.2,
  });

  const notes = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const cost =
    inputTokens * INPUT_COST_PER_TOKEN + outputTokens * OUTPUT_COST_PER_TOKEN;

  console.log(
    `  Synthesis: ${inputTokens} in / ${outputTokens} out ($${cost.toFixed(4)})`
  );

  return { notes, cost };
}

// ── Utils ──

function chunk<T>(arr: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size));
  }
  return result;
}
