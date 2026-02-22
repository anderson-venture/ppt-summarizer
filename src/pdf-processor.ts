import mupdf from "mupdf";
import sharp from "sharp";
import fs from "node:fs";
import path from "node:path";
import { CONFIG } from "./config.js";

export interface ExtractedImage {
  id: string;
  pageNumber: number;
  buffer: Buffer;
  filename: string;
  width: number;
  height: number;
  byteLength: number;
}

export interface PageData {
  pageNumber: number;
  text: string;
  images: ExtractedImage[];
}

export interface ProcessedPDF {
  pages: PageData[];
  totalImages: number;
}

const MIN_IMAGE_DIMENSION = 50;

export async function processPDF(pdfPath: string): Promise<ProcessedPDF> {
  const fileData = fs.readFileSync(pdfPath);
  const doc = mupdf.Document.openDocument(fileData, "application/pdf");
  const pageCount = doc.countPages();

  console.log(`[pdf] Processing ${pageCount} pages...`);

  fs.mkdirSync(CONFIG.imagesDir, { recursive: true });

  const pages: PageData[] = [];
  let totalImages = 0;
  const seenImageHashes = new Set<string>();

  for (let i = 0; i < pageCount; i++) {
    const page = doc.loadPage(i);
    const pageNum = i + 1;

    const stext = page.toStructuredText("preserve-images,preserve-whitespace");
    const text = stext.asText().trim();

    const images: ExtractedImage[] = [];

    stext.walk({
      onImageBlock(_bbox, _transform, image) {
        const w = image.getWidth();
        const h = image.getHeight();
        if (w < MIN_IMAGE_DIMENSION || h < MIN_IMAGE_DIMENSION) return;

        const pixmap = image.toPixmap();
        const pngData = pixmap.asPNG();

        // Deduplicate identical images across pages (e.g. backgrounds)
        const hash = simpleHash(pngData);
        if (seenImageHashes.has(hash)) return;
        seenImageHashes.add(hash);

        totalImages++;
        const id = `img-${pageNum}-${images.length + 1}`;
        const buf = Buffer.from(pngData);
        images.push({
          id,
          pageNumber: pageNum,
          buffer: buf,
          filename: `${id}.jpg`,
          width: w,
          height: h,
          byteLength: buf.length,
        });
      },
    });

    pages.push({ pageNumber: pageNum, text, images });
  }

  // Resize and convert to JPEG to reduce LLM payload and save to disk
  for (const pg of pages) {
    for (const img of pg.images) {
      const pipeline = sharp(img.buffer);
      if (img.width > CONFIG.imageMaxWidth) {
        pipeline.resize({ width: CONFIG.imageMaxWidth });
      }
      img.buffer = await pipeline
        .jpeg({ quality: CONFIG.imageJpegQuality })
        .toBuffer();
      const meta = await sharp(img.buffer).metadata();
      img.width = meta.width ?? img.width;
      img.height = meta.height ?? img.height;
      img.byteLength = img.buffer.length;
      fs.writeFileSync(path.join(CONFIG.imagesDir, img.filename), img.buffer);
    }
  }

  const pagesWithImages = pages.filter((p) => p.images.length > 0).length;
  const pagesWithText = pages.filter((p) => p.text.length > 0).length;
  console.log(
    `[pdf] Done: ${totalImages} images from ${pagesWithImages} pages, ` +
      `text on ${pagesWithText} pages`
  );

  return { pages, totalImages };
}

function simpleHash(data: Uint8Array): string {
  let h = 0;
  const step = Math.max(1, Math.floor(data.length / 1024));
  for (let i = 0; i < data.length; i += step) {
    h = (Math.imul(31, h) + data[i]) | 0;
  }
  return `${h}-${data.length}`;
}
