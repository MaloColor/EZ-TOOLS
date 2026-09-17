// Standalone diagnostic: talks directly to the real Supabase project,
// bypassing the deployed frontend entirely, so we can see ground truth
// regardless of any browser cache / stale-deployment question.
//
// Usage:
//   SUPABASE_URL=... SUPABASE_ANON_KEY=... OUTPUT_PREFIX=sequence_xxx_dvsafe node inspect-real-bucket.mjs
//
// Mirrors the exact logic in file-upload-platform/src/lib/downloadZip.ts:
// reads _complete.json, pages through list() fully, reports any mismatch
// with the specific missing frame indices.

import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;
const OUTPUT_PREFIX = process.env.OUTPUT_PREFIX;
const OUTPUT_BUCKET = process.env.OUTPUT_BUCKET || "depth-outputs";

if (!SUPABASE_URL || !SUPABASE_ANON_KEY || !OUTPUT_PREFIX) {
  console.error("Missing required env vars: SUPABASE_URL, SUPABASE_ANON_KEY, OUTPUT_PREFIX");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
const LIST_PAGE_SIZE = 1000;
const FRAME_NAME_RE = /^frame_(\d+)\.png$/;

async function main() {
  console.log(`Bucket: ${OUTPUT_BUCKET}`);
  console.log(`Prefix: ${OUTPUT_PREFIX}`);
  console.log("");

  // 1. Manifest
  console.log("=== _complete.json ===");
  const manifestPath = `${OUTPUT_PREFIX}/_complete.json`;
  const { data: manifestBlob, error: manifestErr } = await supabase.storage
    .from(OUTPUT_BUCKET)
    .download(manifestPath);
  if (manifestErr) {
    console.log(`Could not download ${manifestPath}:`, manifestErr.message);
  } else {
    const text = await manifestBlob.text();
    console.log(text);
  }
  console.log("");

  // 2. Full paginated listing (same logic as listOutputFrameFiles)
  console.log("=== Paginated list() ===");
  const allFiles = [];
  let offset = 0;
  for (;;) {
    const { data: page, error: listErr } = await supabase.storage
      .from(OUTPUT_BUCKET)
      .list(OUTPUT_PREFIX, { limit: LIST_PAGE_SIZE, offset, sortBy: { column: "name", order: "asc" } });
    if (listErr) {
      console.error(`list() error at offset=${offset}:`, listErr.message);
      break;
    }
    if (!page || page.length === 0) break;
    console.log(`  page at offset=${offset}: ${page.length} item(s)`);
    allFiles.push(...page);
    if (page.length < LIST_PAGE_SIZE) break;
    offset += LIST_PAGE_SIZE;
  }
  console.log(`Total objects listed: ${allFiles.length}`);
  console.log("");

  const frames = allFiles
    .map((f) => {
      const m = FRAME_NAME_RE.exec(f.name);
      return m ? { name: f.name, index: parseInt(m[1], 10) } : null;
    })
    .filter((x) => x !== null)
    .sort((a, b) => a.index - b.index);

  const nonFrameFiles = allFiles.filter((f) => !FRAME_NAME_RE.test(f.name)).map((f) => f.name);

  console.log("=== Summary ===");
  console.log(`Frame files (frame_NNNN.png): ${frames.length}`);
  console.log(`Non-frame files: ${nonFrameFiles.length ? nonFrameFiles.join(", ") : "(none)"}`);
  if (frames.length > 0) {
    console.log(`Frame index range: ${frames[0].index} .. ${frames[frames.length - 1].index}`);

    // Find gaps
    const present = new Set(frames.map((f) => f.index));
    const maxIndex = frames[frames.length - 1].index;
    const missing = [];
    for (let i = 0; i <= maxIndex && missing.length < 50; i++) {
      if (!present.has(i)) missing.push(i);
    }
    if (missing.length > 0) {
      console.log(`Gaps found within the range: [${missing.join(", ")}${missing.length >= 50 ? ", ..." : ""}]`);
    } else {
      console.log("No gaps -- frames are contiguous from 0.");
    }
  }
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
