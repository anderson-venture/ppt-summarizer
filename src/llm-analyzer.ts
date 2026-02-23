import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { CONFIG } from "./config.js";
import type { PageData, ExtractedImage } from "./pdf-processor.js";

const COST = {
  "gpt-4o-mini": { input: 0.15 / 1_000_000, output: 0.60 / 1_000_000 },
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
    console.log(
      `[llm] Analyzing ${allImages.length} images using ${CONFIG.model}...`
    );

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

  // ── Step 3: Synthesize study notes ──
  const tStep3 = performance.now();
  console.log("[llm] Synthesizing study notes...");
  const { notes, cost: synthCost } = await synthesizeNotes(
    client,
    synthesisInputPages
  );
  totalCost += synthCost;

  const step3Time = ((performance.now() - tStep3) / 1000).toFixed(1);
  console.log(`  Step 3 took ${step3Time}s (API cost: $${synthCost.toFixed(4)})\n`);
  console.log(`[llm] Total estimated API cost: $${totalCost.toFixed(4)}`);

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

// ── Study note synthesis ──

interface Content {
  contentId: string;
  title: string;
  pages: number[]; // Pages covered by this content
  subcontents?: Content[]; // Optional subcontents (for hierarchical structure)
}

interface BoneStructure {
  summarization: string; // What students will learn
  contents: Content[]; // Contents (like chapters/sections) with page mappings
}

interface ContentNotes {
  contentId: string;
  mainNote: string;
  reviewQuestions: string;
  glossary: string;
  commonPitfalls: string;
}

// Step 1: Build bone structure (flowchart + page mapping)
async function buildBoneStructure(
  client: OpenAI,
  synthesisInputPages: SynthesisPageInput[]
): Promise<{ structure: BoneStructure; cost: number }> {
  // Filter pages with content and format as JSON
  const pagesWithContent = synthesisInputPages.filter(
    (p) => p.text.length > 0 || p.images.length > 0
  );
  const inputJson = JSON.stringify(pagesWithContent, null, 2);

  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: `You are an expert at organizing lecture content into structured contents like chapters and sections in a textbook.`,
    },
    {
      role: "user",
      content: `Analyze this ${synthesisInputPages.length}-slide lecture and create a structured representation.

TASK:
1. Write a 100-word summarization of what students will learn (key learning objectives).

2. Organize content into contents (like textbook chapters/sections):
   - Each content has a clear title and covers logical page groupings
   - Contents can have subcontents for hierarchical organization
   - Maintain page order (contents follow page sequence)

CRITICAL RULES:
- ALL pages must be covered by exactly one content (no overlap between same-level contents)
- If a content has subcontents, subcontent pages are a subset of parent pages
- Subcontents within the same parent must not overlap

OUTPUT FORMAT (JSON):
{
  "summarization": "What students will learn (approximately 100 words)",
  "contents": [
    {
      "contentId": "content1",
      "title": "Content Title (like a chapter name)",
      "pages": [1, 2, 3, 4, 5],
      "subcontents": [
        {
          "contentId": "content1-1",
          "title": "Subcontent Title (like a section name)",
          "pages": [2, 3]
        },
        {
          "contentId": "content1-2",
          "title": "Another Subcontent Title",
          "pages": [4, 5]
        }
      ]
    },
    {
      "contentId": "content2",
      "title": "Another Content Title",
      "pages": [6, 7, 8]
    },
    ...
  ]
}

--- INPUT PAGES (JSON) ---
${inputJson}`,
    },
  ];

  const resp = await client.chat.completions.create({
    model: CONFIG.model,
    messages,
    max_tokens: 4000,
    temperature: 0,
    response_format: { type: "json_object" },
  });

  const text = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const rates = COST[CONFIG.model];
  const cost = inputTokens * rates.input + outputTokens * rates.output;

  console.log(
    `  Step 1 (Bone Build): ${inputTokens} in / ${outputTokens} out ($${cost.toFixed(4)})`
  );

  let structure: BoneStructure;
  try {
    structure = JSON.parse(text);
  } catch (e) {
    throw new Error(`Failed to parse bone structure JSON: ${e}`);
  }

  // Helper function to collect all pages from a content (including subcontents)
  function collectAllPages(content: Content): number[] {
    const pages = [...content.pages];
    if (content.subcontents) {
      for (const subcontent of content.subcontents) {
        pages.push(...collectAllPages(subcontent));
      }
    }
    return pages;
  }

  // Validate that all pages are matched and there's no overlap
  const allPageNumbers = new Set(
    synthesisInputPages
      .filter((p) => p.text.length > 0 || p.images.length > 0)
      .map((p) => p.pageNumber)
  );
  
  const allMatchedPages: number[] = [];
  for (const content of structure.contents) {
    allMatchedPages.push(...collectAllPages(content));
  }
  const matchedPages = new Set(allMatchedPages);
  
  const unmatchedPages = Array.from(allPageNumbers).filter(
    (p) => !matchedPages.has(p)
  );
  if (unmatchedPages.length > 0) {
    console.warn(
      `Warning: Some pages are not matched to any content: ${unmatchedPages.join(", ")}`
    );
  }

  // Check for overlapping pages (only between same-level contents)
  // 1. Check overlaps between top-level contents
  const topLevelPageMap = new Map<number, string>();
  for (const content of structure.contents) {
    for (const pageNum of content.pages) {
      if (topLevelPageMap.has(pageNum)) {
        console.warn(
          `Warning: Page ${pageNum} is assigned to multiple top-level contents: "${topLevelPageMap.get(pageNum)}" and "${content.title}"`
        );
      } else {
        topLevelPageMap.set(pageNum, content.title);
      }
    }
  }

  // 2. Check overlaps between subcontents within each parent content
  for (const content of structure.contents) {
    if (content.subcontents && content.subcontents.length > 0) {
      const subcontentPageMap = new Map<number, string>();
      for (const subcontent of content.subcontents) {
        for (const pageNum of subcontent.pages) {
          if (subcontentPageMap.has(pageNum)) {
            console.warn(
              `Warning: Page ${pageNum} is assigned to multiple subcontents within "${content.title}": "${subcontentPageMap.get(pageNum)}" and "${subcontent.title}"`
            );
          } else {
            subcontentPageMap.set(pageNum, subcontent.title);
          }
        }
      }
    }
  }

  return { structure, cost };
}

// Step 2: Get notes for each bone (parallel processing)
async function getNotesForEachBone(
  client: OpenAI,
  contents: Content[],
  synthesisInputPages: SynthesisPageInput[]
): Promise<{ contentNotes: ContentNotes[]; cost: number }> {
  const pageMap = new Map(synthesisInputPages.map((p) => [p.pageNumber, p]));

  const contentPromises = contents.map(async (content): Promise<{ notes: ContentNotes; cost: number }> => {
    // Get pages for this content
    const contentPages = content.pages
      .map((pageNum) => pageMap.get(pageNum))
      .filter((p): p is SynthesisPageInput => p !== undefined)
      .filter((p) => p.text.length > 0 || p.images.length > 0);

    // Format pages as JSON
    const pagesJson = JSON.stringify(contentPages, null, 2);

    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: `You are an expert at creating engaging, descriptive study notes that help students understand and remember concepts. Write in a narrative, explanatory style that connects ideas and explains the "why" behind concepts, not just the "what".`,
      },
      {
        role: "user",
        content: `Create engaging study notes for this section of the lecture (Content: "${content.title}").

WRITING STYLE:
- Write descriptively and engagingly — explain concepts with context and connections
- Use narrative explanations where helpful — tell the story behind the concepts
- Connect ideas with smooth transitions — show how concepts relate to each other
- Explain the "why" behind concepts, not just facts — help students understand reasoning
- Use markdown: ## major topics, ### subtopics. Mix short paragraphs with bullets
- Include key definitions, processes, and classifications — but explain their significance
- IMAGES: Add images ONLY where critical for understanding. Embed as ![caption](FILENAME) inline
- Use **bold** for key terms. Mark exam-critical points with ⭐

Make the notes interesting and engaging to read for longer study sessions.

OUTPUT FORMAT (JSON):
{
  "mainNote": "Descriptive study note content in markdown format",
  "reviewQuestions": "**Q:** ...\n **A:** ...\n(if applicable)",
  "glossary": "- **Term**: Definition\n(if applicable)",
  "commonPitfalls": "- Misconception: Correction\n(if applicable)"
}

--- INPUT PAGES FOR THIS CONTENT (JSON) ---
${pagesJson}`,
      },
    ];

    const resp = await client.chat.completions.create({
      model: CONFIG.model,
      messages,
      max_tokens: 3000,
      temperature: 0,
      response_format: { type: "json_object" },
    });

    const text = resp.choices[0]?.message?.content ?? "";
    const inputTokens = resp.usage?.prompt_tokens ?? 0;
    const outputTokens = resp.usage?.completion_tokens ?? 0;
    const rates = COST[CONFIG.model];
    const cost = inputTokens * rates.input + outputTokens * rates.output;

    let notes: Omit<ContentNotes, "contentId">;
    try {
      notes = JSON.parse(text);
    } catch (e) {
      throw new Error(
        `Failed to parse notes JSON for content ${content.contentId}: ${e}`
      );
    }

    return {
      notes: {
        contentId: content.contentId,
        ...notes,
      },
      cost,
    };
  });

  const results = await Promise.all(contentPromises);
  const contentNotes = results.map((r) => r.notes);
  const totalCost = results.reduce((sum, r) => sum + r.cost, 0);

  console.log(
    `  Step 2 (Bone Notes): Processed ${contents.length} contents in parallel ($${totalCost.toFixed(4)})`
  );

  return { contentNotes, cost: totalCost };
}

// Generate Mermaid flowchart based on content structure
async function generateMermaidFlowchart(
  client: OpenAI,
  structure: BoneStructure
): Promise<{ mermaidFlowchart: string; cost: number }> {
  const structureJson = JSON.stringify(
    {
      contents: structure.contents,
    },
    null,
    2
  );

  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: `You are an expert at creating Mermaid flowcharts. Generate ONLY valid Mermaid code.`,
    },
    {
      role: "user",
      content: `Generate Mermaid flowchart code for the following content structure.

--- CONTENT STRUCTURE (JSON) ---
${structureJson}

OUTPUT FORMAT (JSON):
{
  "mermaidFlowchart": "flowchart TD\\n    A[Content Title]\\n    B[Another Content]\\n    A --> B"
}

CRITICAL REQUIREMENTS:
- The mermaidFlowchart value must be EXACTLY the Mermaid code starting with "flowchart TD" (or another valid diagram type)
- Do NOT include the word "mermaidFlowchart" anywhere in the code
- Do NOT include markdown code blocks (\`\`\`mermaid or \`\`\`)
- Do NOT include any prefix, suffix, or labels
- The code must start directly with the diagram type (e.g., "flowchart TD")
- Use \\n for line breaks in the JSON string

CORRECT EXAMPLE:
"mermaidFlowchart": "flowchart TD\\n    A[Title]\\n    B[Title2]\\n    A --> B"

WRONG EXAMPLES (DO NOT DO THIS):
- "mermaidFlowchart flowchart TD..." (has "mermaidFlowchart" prefix)
- "mermaidFlowchart: flowchart TD..." (has label)
- "\`\`\`mermaid\\nflowchart TD..." (has markdown blocks)`,
    },
  ];

  const resp = await client.chat.completions.create({
    model: CONFIG.model,
    messages,
    max_tokens: 2000,
    temperature: 0,
    response_format: { type: "json_object" },
  });

  const text = resp.choices[0]?.message?.content ?? "";
  const inputTokens = resp.usage?.prompt_tokens ?? 0;
  const outputTokens = resp.usage?.completion_tokens ?? 0;
  const rates = COST[CONFIG.model];
  const cost = inputTokens * rates.input + outputTokens * rates.output;

  console.log(
    `  Mermaid Generation: ${inputTokens} in / ${outputTokens} out ($${cost.toFixed(4)})`
  );

  let result: { mermaidFlowchart: string };
  try {
    result = JSON.parse(text);
  } catch (e) {
    throw new Error(`Failed to parse mermaid flowchart JSON: ${e}`);
  }

  return { mermaidFlowchart: result.mermaidFlowchart, cost };
}

// Step 3: Build full note by combining all bones
function buildFullNote(
  structure: BoneStructure,
  contentNotes: ContentNotes[],
  mermaidFlowchart: string
): string {

  const contentMap = new Map(contentNotes.map((n) => [n.contentId, n]));

  // Helper function to format contents structure
  function formatContents(contents: Content[], level: number = 0): string[] {
    const parts: string[] = [];
    const prefix = level === 0 ? "##" : level === 1 ? "###" : "####";
    
    for (const content of contents) {
      parts.push(`${prefix} ${content.title}\n`);
      
      if (content.subcontents && content.subcontents.length > 0) {
        parts.push(...formatContents(content.subcontents, level + 1));
      }
    }
    
    return parts;
  }

  // Build main note by filling each content with their main note
  // Follow the content structure to maintain order
  const mainNoteParts: string[] = [];

  // Use the order in structure.contents which maintains page order
  for (const content of structure.contents) {
    const notes = contentMap.get(content.contentId);
    if (notes) {
      mainNoteParts.push(`${notes.mainNote}\n`);
    }
  }

  const mainNote = mainNoteParts.join("\n");

  // Combine all additional sections
  const allReviewQuestions = contentNotes
    .map((n) => n.reviewQuestions)
    .filter((q) => q.trim().length > 0)
    .join("\n\n");
  const allGlossary = contentNotes
    .map((n) => n.glossary)
    .filter((g) => g.trim().length > 0)
    .join("\n\n");
  const allCommonPitfalls = contentNotes
    .map((n) => n.commonPitfalls)
    .filter((p) => p.trim().length > 0)
    .join("\n\n");

  // Build final output in the requested order
  const parts: string[] = [];

  // 1. Summarization
  parts.push(`# Summary\n\n${structure.summarization}\n\n`);

  // 2. Contents structure
  parts.push(`# Contents\n\n`);
  parts.push(...formatContents(structure.contents));
  parts.push(`\n`);

  // 3. Mermaid Flowchart - ensure it's wrapped in code blocks
  let mermaidCode = mermaidFlowchart.trim();
  parts.push(`# Concept Map\n\n\`\`\`mermaid\n${mermaidCode}\n\`\`\`\n\n`);

  // 4. Combined full main content
  parts.push(`# Study Notes\n\n${mainNote}\n\n`);

  // 5. Combined full review questions
  if (allReviewQuestions) {
    parts.push(`# Review Questions\n\n${allReviewQuestions}\n\n`);
  }

  // 6. Combined full glossary
  if (allGlossary) {
    parts.push(`# Glossary\n\n${allGlossary}\n\n`);
  }

  // 7. Combined full common pitfalls
  if (allCommonPitfalls) {
    parts.push(`# Common Pitfalls\n\n${allCommonPitfalls}\n\n`);
  }

  return parts.join("");
}

async function synthesizeNotes(
  client: OpenAI,
  synthesisInputPages: SynthesisPageInput[]
): Promise<{ notes: string; cost: number }> {
  let totalCost = 0;

  // Step 1: Build bone structure
  console.log("  [Step 1] Building bone structure (flowchart + page mapping)...");
  const { structure, cost: step1Cost } = await buildBoneStructure(
    client,
    synthesisInputPages
  );
  totalCost += step1Cost;

  // Step 2: Get notes for each bone and generate mermaid flowchart (parallel)
  console.log(
    `  [Step 2] Getting notes for ${structure.contents.length} contents and generating mermaid flowchart (parallel)...`
  );
  const [notesResult, mermaidResult] = await Promise.all([
    getNotesForEachBone(client, structure.contents, synthesisInputPages),
    generateMermaidFlowchart(client, structure),
  ]);
  const { contentNotes } = notesResult;
  const { mermaidFlowchart } = mermaidResult;
  const step2Cost = notesResult.cost + mermaidResult.cost;
  totalCost += step2Cost;

  // Step 3: Build full note
  console.log("  [Step 3] Building full note...");
  const notes = buildFullNote(structure, contentNotes, mermaidFlowchart);

  console.log(
    `  Synthesis complete: Total cost $${totalCost.toFixed(4)}`
  );

  return { notes, cost: totalCost };
}

// ── Utils ──

function chunk<T>(arr: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size));
  }
  return result;
}
