import JSZip from "jszip";
import { supabase } from "./supabaseClient";

const LIST_PAGE_SIZE = 1000; // Supabase storage .list() only returns up to `limit` items per call (default 100)
const DOWNLOAD_CONCURRENCY = 12; // parallel per-frame downloads, not one-at-a-time

/**
 * Lists every file the worker wrote under `outputBucket/outputPrefix`,
 * downloads each blob, and packages them into a single zip the browser saves
 * as `${zipName}.zip` — the worker uploads one file per frame, not one file,
 * so this is what makes the design's single "Download" button truthful.
 */
export async function downloadOutputAsZip(
  outputBucket: string,
  outputPrefix: string,
  zipName: string
): Promise<void> {
  // list() is paginated -- a single call caps out at `limit` (default 100)
  // results, which is why only ~100 of a real sequence's hundreds or
  // thousands of frames were coming back before. Page through with
  // limit/offset until a page comes back short of a full page, which means
  // we've reached the end. sortBy keeps frame order stable across pages.
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
    if (page.length < LIST_PAGE_SIZE) break; // last page
    offset += LIST_PAGE_SIZE;
  }

  // "_complete.json" is a marker the worker writes to signal the job
  // finished -- not part of the actual depth sequence, so it's excluded
  // here rather than ending up bundled into the user's download.
  const files = allFiles.filter((f) => f.name !== "_complete.json");
  if (files.length === 0) {
    throw new Error(`No output files found at ${outputBucket}/${outputPrefix}`);
  }

  const zip = new JSZip();

  // Downloading one frame at a time made total time roughly (per-request
  // latency) x (frame count) -- almost all of that is network round-trip
  // wait, not actual transfer, for a sequence that can run to thousands of
  // small files. A small worker pool overlaps those round trips instead of
  // serializing them. DOWNLOAD_CONCURRENCY is deliberately well under
  // typical HTTP/2-per-host multiplexing limits, so this doesn't need to be
  // pushed higher to be effective.
  let nextIndex = 0;
  async function downloadWorker() {
    while (nextIndex < files.length) {
      const file = files[nextIndex++];
      const path = `${outputPrefix}/${file.name}`;
      const { data: blob, error } = await supabase.storage.from(outputBucket).download(path);
      if (error) throw error;
      zip.file(file.name, blob);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(DOWNLOAD_CONCURRENCY, files.length) }, downloadWorker)
  );

  // JSZip already defaults to STORE (no compression) when `compression`
  // isn't passed, but it's worth being explicit: these frames are already
  // dense pixel data (PNG carries its own internal compression) that gains
  // essentially nothing from a second DEFLATE pass, at real CPU cost across
  // a sequence this size.
  const zipBlob = await zip.generateAsync({ type: "blob", compression: "STORE" });
  const url = URL.createObjectURL(zipBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${zipName}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
