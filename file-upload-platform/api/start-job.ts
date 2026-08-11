import type { VercelRequest, VercelResponse } from "@vercel/node";

// Keeps RUNPOD_API_KEY server-side only — never exposed to the browser.
export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const { RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID } = process.env;
  if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
    res.status(500).json({ error: "Server missing RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID" });
    return;
  }

  const { input_bucket, video_key, output_bucket, output_prefix } = req.body ?? {};
  if (!video_key) {
    res.status(400).json({ error: "video_key is required" });
    return;
  }

  const runpodRes = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: { input_bucket, video_key, output_bucket, output_prefix },
    }),
  });

  if (!runpodRes.ok) {
    const text = await runpodRes.text();
    res.status(runpodRes.status).json({ error: `RunPod error: ${text}` });
    return;
  }

  const data = await runpodRes.json();
  res.status(200).json({ id: data.id, status: data.status });
}
