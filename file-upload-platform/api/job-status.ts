import type { VercelRequest, VercelResponse } from "@vercel/node";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "GET") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const { RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID } = process.env;
  if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
    res.status(500).json({ error: "Server missing RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID" });
    return;
  }

  const id = req.query.id;
  if (!id || typeof id !== "string") {
    res.status(400).json({ error: "id query param is required" });
    return;
  }

  const runpodRes = await fetch(
    `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${id}`,
    { headers: { Authorization: `Bearer ${RUNPOD_API_KEY}` } }
  );

  if (!runpodRes.ok) {
    const text = await runpodRes.text();
    res.status(runpodRes.status).json({ error: `RunPod error: ${text}` });
    return;
  }

  const data = await runpodRes.json();
  // Pass through RunPod's own status vocabulary (IN_QUEUE, IN_PROGRESS,
  // COMPLETED, FAILED) — the client maps these to UI steps itself.
  res.status(200).json({ id: data.id, status: data.status, output: data.output ?? null });
}
