import JSZip from "jszip";
import { supabase } from "./supabaseClient";

const LIST_PAGE_SIZE = 1000;
const DOWNLOAD_CONCURRENCY = 12;
const FRAME_NAME_RE = /^frame_(\d+)\.png$/;

export interface OutputManifest {
  frame_count: number;
  format?: string;
  davinci_safe?: boolean;
  completed_at?: number;
}

/** Worker-written manifest under the output prefix. */
export async function readOutputManifest(
  outputBucket: string,
  outputPrefix: string
): Promise<OutputManifest> {
  const path = `${outputPrefix}/_complete.json`;
  const { data, error } = await supabase.storage.from(outputBucket).download(path);
  if (error) {
    throw new Error(
      "Output isn't ready for download (missing _complete.json). The job may still be running or failed partway through."
    );
  }
  let manifest: OutputManifest;
  try {
    manifest = JSON.parse(await data.text()) as OutputManifest;
  } catch {
    throw new Error("Invalid _complete.json in storage.");
  }
  if (!Number.isInteger(manifest.frame_count) || manifest.frame_count < 1) {
    throw new Error("Invalid _complete.json: frame_count must be a positive integer.");
  }
  console.log(`[download-zip] manifest at ${outputPrefix}/_complete.json:`, manifest);
  return manifest;
}

/** Every frame_*.png the worker wrote under this output prefix (paginated list). */
async function listOutputFrameFiles(
  outputBucket: string,
  outputPrefix: string
): Promise<{ name: string; index: number }[]> {
  const allFiles: { name: string }[] = [];
  let offset = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { data: page, error: listError } = await supabase.storage
      .from(outputBucket)
      .list(outputPrefix, {
        limit: LIST_PAGE_SIZE,
        offset,
        sortBy: { column: "name", order: "asc" },
      });
    if (listError) throw listError;
    if (!page || page.length === 0) break;
    allFiles.push(...page);
    console.log(
      `[download-zip] list(${outputPrefix}) page at offset=${offset}: ${page.length} item(s) (running total ${allFiles.length})`
    );
    if (page.length < LIST_PAGE_SIZE) break;
    offset += LIST_PAGE_SIZE;
  }

  const frames = allFiles
    .map((f) => {
      const m = FRAME_NAME_RE.exec(f.name);
      return m ? { name: f.name, index: parseInt(m[1], 10) } : null;
    })
    .filter((x): x is { name: string; index: number } => x !== null)
    .sort((a, b) => a.index - b.index);

  const nonFrameFiles = allFiles.filter((f) => !FRAME_NAME_RE.test(f.name)).map((f) => f.name);
  console.log(
    `[download-zip] ${outputPrefix}: ${allFiles.length} object(s) listed, ${frames.length} match frame_NNNN.png` +
      (nonFrameFiles.length ? `, ${nonFrameFiles.length} non-frame file(s): ${nonFrameFiles.join(", ")}` : "")
  );
  if (frames.length > 0) {
    console.log(
      `[download-zip] frame index range: ${frames[0].index}..${frames[frames.length - 1].index}`
    );
  }
  return frames;
}

/**
 * Ensures storage output matches the completed job manifest before zipping —
 * same files, same count, contiguous frame_0000 … frame_NNNN sequence.
 */
export function assertOutputMatchesManifest(
  frames: { name: string; index: number }[],
  frame_count: number
): void {
  if (frames.length !== frame_count) {
    // Which specific indices are missing, not just the count -- "98 of 300,
    // missing 98..299" (worker stopped early) reads very differently from
    // scattered gaps (a few uploads that failed mid-job), and this is the
    // one place that can tell the two apart.
    const present = new Set(frames.map((f) => f.index));
    const missing: number[] = [];
    for (let i = 0; i < frame_count && missing.length < 20; i++) {
      if (!present.has(i)) missing.push(i);
    }
    console.error(
      `[download-zip] MISMATCH: storage has ${frames.length} frame(s), manifest says ${frame_count}.`,
      `First missing indices: [${missing.join(", ")}${missing.length >= 20 ? ", ..." : ""}]`
    );
    throw new Error(
      `Job output has ${frames.length} frame file(s) in storage but _complete.json says ${frame_count}. Re-run processing to refresh the output.`
    );
  }
  for (let i = 0; i < frame_count; i++) {
    const expected = `frame_${String(i).padStart(4, "0")}.png`;
    if (frames[i].index !== i || frames[i].name !== expected) {
      console.error(
        `[download-zip] ORDER MISMATCH at position ${i}: expected ${expected}, found`,
        frames[i]
      );
      throw new Error(
        `Job output is missing or out of order at ${expected}. Re-run processing to refresh the output.`
      );
    }
  }
  console.log(`[download-zip] verified: ${frame_count} frame(s) match manifest exactly.`);
}

/** Manifest + storage check; use for UI before offering download. */
export async function getVerifiedOutputFrameCount(
  outputBucket: string,
  outputPrefix: string
): Promise<number> {
  const manifest = await readOutputManifest(outputBucket, outputPrefix);
  const frames = await listOutputFrameFiles(outputBucket, outputPrefix);
  assertOutputMatchesManifest(frames, manifest.frame_count);
  return manifest.frame_count;
}

// Downloading frames dominates wall-clock time (network-bound, hundreds to
// thousands of requests); packaging is fast since generateAsync uses STORE
// (no compression). Weighting the two phases like this means the bar moves
// smoothly through the slow part instead of sitting at 0% then jumping to
// 100% once the last frame lands.
const DOWNLOAD_PROGRESS_WEIGHT = 0.9;
const ZIP_PROGRESS_WEIGHT = 1 - DOWNLOAD_PROGRESS_WEIGHT;

/**
 * Reads the completed job manifest, verifies every listed output frame exists
 * in storage, then zips exactly those PNGs (nothing else from the bucket).
 * onProgress, if given, is called with a fraction from 0 to 1.
 */
export async function downloadOutputAsZip(
  outputBucket: string,
  outputPrefix: string,
  zipName: string,
  onProgress?: (fraction: number) => void
): Promise<void> {
  const manifest = await readOutputManifest(outputBucket, outputPrefix);
  const frames = await listOutputFrameFiles(outputBucket, outputPrefix);
  assertOutputMatchesManifest(frames, manifest.frame_count);

  const zip = new JSZip();
  let nextIndex = 0;
  let downloadedCount = 0;

  async function downloadWorker() {
    while (nextIndex < frames.length) {
      const index = nextIndex++;
      const file = frames[index];
      const path = `${outputPrefix}/${file.name}`;
      // .download() always attaches an Authorization header, which forces a
      // CORS preflight (OPTIONS) before every single GET -- confirmed in
      // Supabase's own request logs, doubling the request count on jobs
      // with hundreds of frames. depth-outputs is currently a public
      // bucket, so a plain fetch() of its public URL (no custom headers) is
      // a "simple" CORS request and skips the preflight entirely. But
      // getPublicUrl() is just a string builder -- it doesn't check the
      // bucket's actual public/private setting, so if that ever changes
      // (or on a fresh deploy that only follows SETUP.md's RLS-policy
      // instructions without also marking the bucket public), the
      // unauthenticated fetch would fail for every frame. Fall back to the
      // authenticated download in that case so this keeps working either
      // way -- slower without the public bucket, but never broken.
      const {
        data: { publicUrl },
      } = supabase.storage.from(outputBucket).getPublicUrl(path);
      const publicRes = await fetch(publicUrl);
      let blob: Blob;
      if (publicRes.ok) {
        blob = await publicRes.blob();
      } else {
        const { data, error } = await supabase.storage.from(outputBucket).download(path);
        if (error || !data) {
          throw new Error(
            `Could not read ${file.name} from job output (frame ${index + 1} of ${manifest.frame_count}).`
          );
        }
        blob = data;
      }
      zip.file(file.name, blob);
      downloadedCount++;
      onProgress?.((downloadedCount / frames.length) * DOWNLOAD_PROGRESS_WEIGHT);
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(DOWNLOAD_CONCURRENCY, frames.length) }, downloadWorker)
  );

  if (Object.keys(zip.files).length !== manifest.frame_count) {
    throw new Error("Zip did not include every output frame — download aborted.");
  }

  const zipBlob = await zip.generateAsync({ type: "blob", compression: "STORE" }, (metadata) => {
    onProgress?.(DOWNLOAD_PROGRESS_WEIGHT + (metadata.percent / 100) * ZIP_PROGRESS_WEIGHT);
  });
  onProgress?.(1);

  const url = URL.createObjectURL(zipBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${zipName}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoking immediately after click() races the browser's download
  // handler -- for a large blob (hundreds of MB to a few GB on a
  // thousand-plus-frame job) the browser hasn't necessarily started
  // reading it yet, so the URL can go invalid before the save actually
  // begins and the download silently never happens. Give it a few
  // seconds' head start before freeing the blob URL.
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
