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
  const { data: files, error: listError } = await supabase.storage
    .from(outputBucket)
    .list(outputPrefix);
  if (listError) throw listError;
  if (!files || files.length === 0) {
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
