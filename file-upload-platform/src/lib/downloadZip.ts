import JSZip from "jszip";
import { supabase } from "./supabaseClient";

/**
 * Lists every file the worker wrote under `outputBucket/outputPrefix`,
 * downloads each blob, and packages them into a single zip the browser saves
 * as `${zipName}.zip` — the worker uploads one EXR per frame, not one file,
 * so this is what makes the design's single "Download" button truthful.
 */
export async function downloadOutputAsZip(
  outputBucket: string,
  outputPrefix: string,
  zipName: string
): Promise<void> {
  // list() defaults to a 100-item page -- a real sequence can run to
  // thousands of frames, so this has to page through everything instead of
  // taking the first call's result, or a long video would silently zip only
  // its first 100 frames with no error at all.
  const allFiles: { name: string }[] = [];
  const pageSize = 1000;
  for (let offset = 0; ; offset += pageSize) {
    const { data: page, error: listError } = await supabase.storage
      .from(outputBucket)
      .list(outputPrefix, { limit: pageSize, offset });
    if (listError) throw listError;
    if (!page || page.length === 0) break;
    allFiles.push(...page);
    if (page.length < pageSize) break;
  }
  // "_complete.json" is a marker the worker writes to signal the job
  // finished -- not part of the actual depth sequence, so it's excluded
  // here rather than ending up bundled into the user's download.
  const files = allFiles.filter((f) => f.name !== "_complete.json");
  if (files.length === 0) {
    throw new Error(`No output files found at ${outputBucket}/${outputPrefix}`);
  }

  const zip = new JSZip();
  for (const file of files) {
    const path = `${outputPrefix}/${file.name}`;
    const { data: blob, error } = await supabase.storage.from(outputBucket).download(path);
    if (error) throw error;
    zip.file(file.name, blob);
  }

  const zipBlob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(zipBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${zipName}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
