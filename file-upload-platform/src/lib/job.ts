export type JobStatus = "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | string;

export async function startJob(input: {
  input_bucket: string;
  video_key: string;
  output_bucket: string;
  output_prefix: string;
}): Promise<{ id: string }> {
  const res = await fetch("/api/start-job", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `start-job failed (${res.status})`);
  }
  return res.json();
}

export async function getJobStatus(
  id: string
): Promise<{ status: JobStatus; output: unknown }> {
  const res = await fetch(`/api/job-status?id=${encodeURIComponent(id)}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `job-status failed (${res.status})`);
  }
  return res.json();
}

/**
 * Polls RunPod until the job leaves IN_QUEUE/IN_PROGRESS. Calls `onTick` on
 * every poll so the caller can drive step UI off the raw status.
 */
export async function pollJobUntilDone(
  id: string,
  onTick: (status: JobStatus) => void,
  { intervalMs = 3000, timeoutMs = 15 * 60 * 1000 } = {}
): Promise<unknown> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const { status, output } = await getJobStatus(id);
    onTick(status);
    if (status === "COMPLETED") return output;
    if (status === "FAILED" || status === "CANCELLED" || status === "TIMED_OUT") {
      throw new Error(`RunPod job ended with status ${status}`);
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("Timed out waiting for RunPod job to complete");
}
