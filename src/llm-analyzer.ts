import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { CONFIG } from "./config.js";
import type { PageData, ExtractedImage } from "./pdf-processor.js";

const VISION_TOKENS_PER_LOW_IMAGE = 2833;

const COST = {
  "gpt-4o-mini": { input: 0.15 / 1_000_000, output: 0.60 / 1_000_000 },
  "gpt-4o":      { input: 2.50 / 1_000_000, output: 10.0 / 1_000_000 },
} as const;

type ModelKey = keyof typeof COST;

export interface ImageDescription {
  imageId: string;
  pageNumber: number;
  description: string;
}

/** Per-page input fed into the synthesis step (text + image descriptions) */
export interface SynthesisPageInput {
  pageNumber: number;
  text: string;
  images: { filename: string; imageId: string; description: string }[];
}

export interface AnalysisResult {
  imageDescriptions: ImageDescription[];
  synthesisInputPages: SynthesisPageInput[];
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
  const tStep2 = performance.now();

  if (allImages.length > 0) {
    const isComplex = (img: ImageWithContext) =>
      img.width > CONFIG.complexImageMinWidth ||
      img.byteLength > CONFIG.complexImageMinBytes;

    const simpleImages  = allImages.filter((img) => !isComplex(img));
    const complexImages = allImages.filter(isComplex);

    console.log(
      `[llm] Analyzing ${allImages.length} images ` +
        `(${simpleImages.length} simple → ${CONFIG.model}, ` +
        `${complexImages.length} complex → ${CONFIG.premiumModel})...`
    );

    const simpleBatches  = chunk(simpleImages, CONFIG.visionBatchSize);
    const complexBatches = chunk(complexImages, CONFIG.visionBatchSize);
    const totalBatches   = simpleBatches.length + complexBatches.length;

    const batchPromises = [
      ...simpleBatches.map((batch, idx) =>
        describeImageBatch(client, batch, idx, totalBatches, CONFIG.model)
      ),
      ...complexBatches.map((batch, idx) =>
        describeImageBatch(
          client,
          batch,
          simpleBatches.length + idx,
          totalBatches,
          CONFIG.premiumModel
        )
      ),
    ];

    const batchResults = await Promise.all(batchPromises);
    for (const result of batchResults) {
      imageDescriptions.push(...result.descriptions);
      totalCost += result.cost;
    }
  } else {
    console.log("[llm] No images found, skipping vision analysis.");
  }

  const step2Time = ((performance.now() - tStep2) / 1000).toFixed(1);
  const step2Cost = totalCost;
  console.log(`  Step 2 took ${step2Time}s (API cost: $${step2Cost.toFixed(4)})\n`);

  // ── Step 3: Synthesize study notes ──
  const tStep3 = performance.now();
  console.log("[llm] Synthesizing study notes...");
  const { notes, cost: synthCost } = await synthesizeNotes(
    client,
    pages,
    imageDescriptions
  );
  totalCost += synthCost;

  const step3Time = ((performance.now() - tStep3) / 1000).toFixed(1);
  console.log(`  Step 3 took ${step3Time}s (API cost: $${synthCost.toFixed(4)})\n`);
  console.log(`[llm] Total estimated API cost: $${totalCost.toFixed(4)}`);

  // Build per-page synthesis input (text + image descriptions)
  const descMap = new Map(imageDescriptions.map((d) => [d.imageId, d]));
  const synthesisInputPages: SynthesisPageInput[] = pages
    .filter((p) => p.text.length > 0 || p.images.length > 0)
    .map((p) => ({
      pageNumber: p.pageNumber,
      text: p.text,
      images: p.images.map((img) => {
        const d = descMap.get(img.id);
        return {
          filename: img.filename,
          imageId: img.id,
          description: d?.description ?? "(no description)",
        };
      }),
    }));

  return {
    imageDescriptions,
    synthesisInputPages,
    studyNotes: notes,
    costEstimate: totalCost,
  };
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
  totalBatches: number,
  model: ModelKey = CONFIG.model
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
        url: `data:image/jpeg;base64,${base64}`,
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
    model,
    messages,
    max_tokens: 3000,
    temperature: 0.2,
  });

  const text = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const rates = COST[model];
  const cost = inputTokens * rates.input + outputTokens * rates.output;

  console.log(
    `  Batch ${batchIdx + 1}/${totalBatches} [${model}]: ${inputTokens} in / ${outputTokens} out ($${cost.toFixed(4)})`
  );

  const descriptions = parseImageDescriptions(text, batch);
  return { descriptions, cost };
}

function buildVisionPrompt(batch: ImageWithContext[]): string {
  const imageList = batch
    .map((img, i) => {
      const ctx = img.pageText
        ? `\n   Slide text: "${img.pageText}"`
        : "";
      return `${i + 1}. Image ID: ${img.id} (page ${img.pageNumber})${ctx}`;
    })
    .join("\n\n");

  return `You are analyzing images extracted from university lecture slides. These are scientific images or diagrams essential for study.

For each image, reason step by step before writing the final description:
1. Identify all visible text, labels, and symbols in the image.
2. Explain what each element represents in the context of the slide text provided.
3. Summarise the diagram in one sentence.

Then output a structured description with these four parts:
- **Main concept**: What the diagram illustrates and why it matters.
- **Key labels**: Important text, terms, or symbols visible in the image.
- **Relationships/processes**: Arrows, flows, steps, hierarchies, or cycles shown.
- **Exam relevance**: What a student must remember about this diagram.

Images to analyze:
${imageList}

The images follow in the same order. Respond in this exact format:

[${batch[0].id}]
<structured description>

[${batch[1]?.id ?? "..."}]
<structured description>

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
      content: `You are an expert at turning university lectures into comprehensive, engaging study notes. Write in a descriptive, narrative style — not just bullet-point facts. Use transitions to connect ideas, explain the "why" behind concepts, and make the notes pleasant to read for longer study sessions. Diagrams are placed INLINE next to the concept they illustrate — never in a separate section.`,
    },
    {
      role: "user",
      content: `Summarize this ${pages.length}-slide lecture into study notes that are thorough AND engaging to read.

CRITICAL FORMATTING RULE:
- Each diagram MUST be placed IMMEDIATELY AFTER the bullet point or paragraph that discusses that concept.
- For example, when you explain transformation, place the transformation diagram right there — NOT in a separate "diagrams" section.
- NEVER create a dedicated images/diagrams section. Images must be woven into the text.

WRITING STYLE:
1. Descriptive and engaging — avoid dry bullet-point fact dumps. Use narrative explanations where helpful.
2. Connect ideas with transitions. Explain the "why" behind concepts, not just the "what."
3. Balance thoroughness with readability. Notes should hold attention during longer study sessions.
4. Use markdown: # title, ## major topics, ### subtopics. Mix short paragraphs with bullets where appropriate. Tables for comparisons.
5. Include key definitions, classifications, numerical facts, and step-by-step processes — but explain their significance.
6. IMAGES: Add images ONLY where they are critical and helpful for understanding the concept — not a fixed count. Quality over quantity. If an image does not directly aid comprehension of that specific topic, omit it. Embed as ![caption](FILENAME) inline immediately after the text that explains that concept. Do NOT add images just to fill a quota; include only diagrams that students genuinely need to see to understand the material.
7. Use **bold** for key terms. Mark exam-critical bullets or headings with a ⭐ at the start. Aim for 1-3 sentences per point so ideas breathe.

ADDITIONAL SECTIONS (append after the main notes, in this order):

## Review Questions
Add 5–8 questions that test the most important concepts from the lecture. Include a brief answer after each question (e.g. "**Q:** … **A:** …").

## Glossary
Define each bold key term introduced in the lecture in one concise sentence.

## Concept Map
Add a Mermaid flowchart showing how the lecture's top 5–7 topics connect. Use \`\`\`mermaid syntax.

## Common Pitfalls
List 3–5 misconceptions students commonly have about this material, with a brief correction for each.

--- SLIDE TEXT ---
${textContent}

--- IMAGE CATALOG (${allImages.length} images available) ---
${imageCatalog}`,
    },
  ];

  const resp = await client.chat.completions.create({
    model: CONFIG.model,
    messages,
    max_tokens: 6000,
    temperature: 0,
  });

  const notes = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const rates = COST[CONFIG.model];
  const cost = inputTokens * rates.input + outputTokens * rates.output;

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
