import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { CONFIG } from "./config.js";
import type { PageData, ExtractedImage } from "./pdf-processor.js";

const VISION_TOKENS_PER_LOW_IMAGE = 2833;

const COST = {
  "gpt-4o-mini": { input: 0.15 / 1_000_000, output: 0.60 / 1_000_000 },
  "gpt-4o":      { input: 2.50 / 1_000_000, output: 5.0 / 1_000_000 },
} as const;

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

/** Topic with page numbers (from Step 1) */
export interface TopicWithPages {
  name: string;
  pageNumbers: number[];
}

/** Step 1 result: topics and mermaid flowchart */
interface TopicsAndFlowchart {
  topics: TopicWithPages[];
  mermaidFlowchart: string;
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
    console.log(`[llm] Analyzing ${allImages.length} images (${CONFIG.model})...`);

    const batches = chunk(allImages, CONFIG.visionBatchSize);
    const totalBatches = batches.length;

    const batchPromises = batches.map((batch, idx) =>
      describeImageBatch(client, batch, idx, totalBatches)
    );

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

  // ── Step 3: Three-step synthesis ──
  const tStep3 = performance.now();
  const descMap = new Map(imageDescriptions.map((d) => [d.imageId, d]));

  // Step 3a: Text-only → topics + mermaid flowchart
  console.log("[llm] Step 3a: Extracting topics and flowchart (text only)...");
  const { topics, mermaidFlowchart, cost: cost3a } =
    await extractTopicsAndFlowchart(client, pages);
  totalCost += cost3a;
  console.log(`  Topics: ${topics.map((t) => t.name).join(", ")}`);

  // Step 3b: Per-topic summarization (text + images for topic pages) — parallel
  console.log("[llm] Step 3b: Summarizing each topic (parallel)...");
  const step3bResults = await Promise.all(
    topics.map((topic) => summarizeTopic(client, topic, pages, descMap))
  );
  const topicSummaries = step3bResults.map((r) => r.summary);
  for (const r of step3bResults) totalCost += r.cost;

  // Step 3c: Merge into final study notes using flowchart order
  console.log("[llm] Step 3c: Merging study notes...");
  const { notes, cost: mergeCost } = await mergeStudyNotes(
    client,
    topics,
    topicSummaries,
    mermaidFlowchart,
    pages
  );
  totalCost += mergeCost;

  const step3Time = ((performance.now() - tStep3) / 1000).toFixed(1);
  console.log(`  Step 3 took ${step3Time}s (API cost: $${(totalCost - step2Cost).toFixed(4)})\n`);
  console.log(`[llm] Total estimated API cost: $${totalCost.toFixed(4)}`);

  // Build per-page synthesis input (text + image descriptions)
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
    model: CONFIG.model,
    messages,
    max_tokens: 3000,
    temperature: 0.2,
  });

  const text = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const rates = COST[CONFIG.model];
  const cost = inputTokens * rates.input + outputTokens * rates.output;

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

// ── Study note synthesis (3 steps) ──

function buildTextContentForPages(pages: PageData[]): string {
  return pages
    .filter((p) => p.text.length > 0 || p.images.length > 0)
    .map((p) => {
      let entry = `[p${p.pageNumber}]`;
      if (p.text) entry += ` ${p.text}`;
      for (const img of p.images) entry += ` {{${img.filename}}}`;
      return entry;
    })
    .join("\n");
}

/** Step 1: Text only → topics with page numbers + mermaid flowchart */
async function extractTopicsAndFlowchart(
  client: OpenAI,
  pages: PageData[]
): Promise<TopicsAndFlowchart & { cost: number }> {
  const textContent = buildTextContentForPages(pages);

  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: `You are an expert at analyzing lecture structure. Given slide text only, identify the main topics and how they connect. Output valid JSON only.`,
    },
    {
      role: "user",
      content: `Analyze this ${pages.length}-slide lecture (text only below).

1. Split the content into main TOPICS. For each topic, list the exact PAGE NUMBERS (from the [pN] markers) that belong to that topic. A page may appear in more than one topic if it spans concepts.
2. Create a Mermaid flowchart that shows how these topics connect (e.g. prerequisite order, logical flow). Use standard Mermaid syntax (flowchart LR or TD, nodes, arrows). The flowchart should help students see the lecture structure at a glance.

Respond with ONLY a single JSON object, no markdown or extra text:
{
  "topics": [
    { "name": "Topic title", "pageNumbers": [1, 2, 3] }
  ],
  "mermaidFlowchart": "flowchart LR\\n  A --> B\\n  B --> C"
}

--- SLIDE TEXT ---
${textContent}`,
    },
  ];

  const resp = await client.chat.completions.create({
    model: CONFIG.model,
    messages,
    max_tokens: 4000,
    temperature: 0,
  });

  const text = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const rates = COST[CONFIG.model];
  const cost = inputTokens * rates.input + outputTokens * rates.output;

  const parsed = parseTopicsAndFlowchart(text);
  return { ...parsed, cost };
}

function parseTopicsAndFlowchart(text: string): TopicsAndFlowchart {
  const trimmed = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
  let data: { topics?: { name: string; pageNumbers: number[] }[]; mermaidFlowchart?: string };
  try {
    data = JSON.parse(trimmed) as typeof data;
  } catch {
    return { topics: [], mermaidFlowchart: "flowchart LR\n  A[Topics] --> B[Summary]" };
  }
  const topics: TopicWithPages[] = (data.topics ?? []).map((t) => ({
    name: String(t.name ?? "Topic"),
    pageNumbers: Array.isArray(t.pageNumbers) ? t.pageNumbers.map(Number) : [],
  }));
  const mermaidFlowchart = typeof data.mermaidFlowchart === "string"
    ? data.mermaidFlowchart.trim()
    : "flowchart LR\n  A[Topics] --> B[Summary]";
  return { topics, mermaidFlowchart };
}

/** Step 2: For one topic, summarize text + image descriptions for its pages */
async function summarizeTopic(
  client: OpenAI,
  topic: TopicWithPages,
  pages: PageData[],
  descMap: Map<string, ImageDescription>
): Promise<{ summary: string; cost: number }> {
  const pageSet = new Set(topic.pageNumbers);
  const topicPages = pages.filter((p) => pageSet.has(p.pageNumber));
  const textContent = topicPages
    .filter((p) => p.text.length > 0 || p.images.length > 0)
    .map((p) => {
      let entry = `[p${p.pageNumber}]`;
      if (p.text) entry += ` ${p.text}`;
      for (const img of p.images) {
        const desc = descMap.get(img.id);
        entry += `\n  {{${img.filename}}}: ${desc?.description ?? "diagram"}`;
      }
      return entry;
    })
    .join("\n\n");

  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: `You are an expert at turning university lecture content into clear, engaging study notes. Write in a descriptive, narrative style. Place diagrams INLINE next to the concept they illustrate (as ![caption](FILENAME)). Use **bold** for key terms. Be thorough but readable.`,
    },
    {
      role: "user",
      content: `Summarize this single topic for study notes.

Topic: **${topic.name}**
Pages covered: ${topic.pageNumbers.join(", ")}

RULES:
- Write a ## ${topic.name} section with subsections (###) as needed.
- Weave in image references where they help: use ![brief caption](FILENAME) right after the sentence that explains that concept. Only include images that are critical for understanding.
- Descriptive and engaging; explain the "why" behind concepts. Use transitions between ideas.
- Use markdown: **bold** for key terms, bullets and short paragraphs.

--- CONTENT FOR THIS TOPIC ---
${textContent}`,
    },
  ];

  const resp = await client.chat.completions.create({
    model: CONFIG.model,
    messages,
    max_tokens: 4000,
    temperature: 0,
  });

  const summary = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const rates = COST[CONFIG.model];
  const cost = inputTokens * rates.input + outputTokens * rates.output;

  return { summary, cost };
}

/** Step 3: Merge topic summaries using the flowchart order and add closing sections */
async function mergeStudyNotes(
  client: OpenAI,
  topics: TopicWithPages[],
  topicSummaries: string[],
  mermaidFlowchart: string,
  pages: PageData[]
): Promise<{ notes: string; cost: number }> {
  const topicBlocks = topics
    .map((t, i) => `### ${t.name}\n${topicSummaries[i] ?? ""}`)
    .join("\n\n");

  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: `You are an expert at assembling study notes. Merge topic sections into one coherent document. Preserve the exact Mermaid diagram provided. Add a title, intro, and closing sections (Review Questions, Glossary, Common Pitfalls). Write in a descriptive, engaging style.`,
    },
    {
      role: "user",
      content: `Merge these topic-based sections into one final study note document.

STRUCTURE (follow this order):
1. # [Lecture Title] — choose a concise title from the content.
2. Brief intro paragraph (2–4 sentences) that sets the scope of the lecture.
3. ## Concept Map — paste the Mermaid flowchart exactly as given below (in \`\`\`mermaid ... \`\`\`).
4. ## Main content — merge the topic sections below in the SAME order as in the flowchart. Use ## for each topic heading. Keep all inline images and formatting from each topic summary.
5. ## Review Questions — add 5–8 questions with brief answers (**Q:** … **A:** …).
6. ## Glossary — one-sentence definitions for key bold terms from the notes.
7. ## Common Pitfalls — 3–5 common misconceptions with brief corrections.

MERMAID FLOWCHART (use exactly):
\`\`\`mermaid
${mermaidFlowchart}
\`\`\`

TOPIC SECTIONS TO MERGE (in flowchart order):
${topicBlocks}

Produce the full study notes as a single markdown document.`,
    },
  ];

  const resp = await client.chat.completions.create({
    model: CONFIG.model,
    messages,
    max_tokens: 10000,
    temperature: 0,
  });

  const notes = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const rates = COST[CONFIG.model];
  const cost = inputTokens * rates.input + outputTokens * rates.output;

  console.log(
    `  Merge: ${inputTokens} in / ${outputTokens} out ($${cost.toFixed(4)})`
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
