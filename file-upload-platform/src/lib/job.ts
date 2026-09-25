export type JobStatus = "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | string;

export async function startJob(input: {
  input_bucket: string;
  video_key: string;
  output_bucket: string;
  output_prefix: string;
  davinci_safe: boolean;
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
): Promise<{ status: JobStatus; output: unknown; error?: string | null }> {
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
 *
 * `output` is passed through as-is while the job is still running: the
 * worker can push intermediate values via RunPod's progress_update() API
 * (see handler.py's on_progress), which land in this exact field with
 * status still IN_PROGRESS -- so a worker that reports progress lets the
 * caller show real frame-based percentage; an older worker that doesn't
 * just means output stays null until the job actually completes, which
 * callers need to handle gracefully.
 */
export async function pollJobUntilDone(
  id: string,
  onTick: (status: JobStatus, output: unknown) => void,
  { intervalMs = 3000, timeoutMs = 15 * 60 * 1000 } = {}
): Promise<unknown> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const { status, output, error } = await getJobStatus(id);
    onTick(status, output);
    if (status === "COMPLETED") return output;
    if (status === "FAILED" || status === "CANCELLED" || status === "TIMED_OUT") {
      // `error` is the worker's actual failure reason (see handler.py's
      // except block) when RunPod reports one -- surface it instead of
      // just the status, so a real crash (bad video, OOM, network error)
      // shows up as something the user or you can actually act on.
      throw new Error(error ? `Processing failed: ${error}` : `RunPod job ended with status ${status}`);
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("Timed out waiting for RunPod job to complete");
}
