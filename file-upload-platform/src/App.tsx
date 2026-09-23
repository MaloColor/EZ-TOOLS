import { useRef, useState, type CSSProperties } from "react";
import { supabase, isSupabaseConfigured, INPUT_BUCKET, OUTPUT_BUCKET } from "./lib/supabaseClient";
import { startJob, pollJobUntilDone, type JobStatus } from "./lib/job";
import { downloadOutputAsZip, getVerifiedOutputFrameCount } from "./lib/downloadZip";

type View = "idle" | "configuring" | "processing" | "done" | "error";
type Overlay = "none" | "about" | "login" | "settings";

const MAX_BYTES = 100 * 1024 * 1024;
const MAX_DURATION_SECONDS = 60;
const OUTPUT_FORMAT_LABEL = "16-BIT PNG Depth Sequence";
const STEP_LABELS = ["Uploading", "Analyzing", "Preparing output"];

function formatSize(bytes: number): string {
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(0)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// Reads a video file's length by loading its metadata into an off-DOM
// <video> element -- File objects carry no duration themselves, this is
// the only way to get it without uploading the file anywhere first.
function getVideoDuration(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    const url = URL.createObjectURL(file);
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(video.duration);
    };
    video.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Couldn't read this file's video metadata."));
    };
    video.src = url;
  });
}

function sanitizeFileName(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]/g, "_");
}

// Content hash of the file, used as a stable key so re-uploading the same
// video (with the same DaVinci setting) maps to the same output location
// instead of always kicking off a fresh RunPod job.
async function sha256Hex(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

interface OutputInfo {
  prefix: string;
  baseName: string;
  frameCount: number;
}

async function outputInfoForJob(prefix: string, baseName: string): Promise<OutputInfo> {
  const frameCount = await getVerifiedOutputFrameCount(OUTPUT_BUCKET, prefix);
  return { prefix, baseName, frameCount };
}

interface RenderProgress {
  frameIndex: number;
  totalFrames: number;
}

// The worker pushes {frame_index, total_frames} via RunPod's progress_update()
// API while IN_PROGRESS (see handler.py's on_progress) -- that lands verbatim
// in the job-status response's `output` field. An older, not-yet-redeployed
// worker never sends this, so `output` stays null/unrecognized the whole
// time; callers need to fall back gracefully rather than assume it's there.
function parseRenderProgress(output: unknown): RenderProgress | null {
  if (!output || typeof output !== "object") return null;
  const o = output as Record<string, unknown>;
  if (
    typeof o.frame_index === "number" &&
    typeof o.total_frames === "number" &&
    o.total_frames > 0 &&
    o.frame_index >= 0
  ) {
    return { frameIndex: o.frame_index, totalFrames: o.total_frames };
  }
  return null;
}

// Estimates seconds remaining from how many frames got done between the
// first progress sample we saw and this one -- a live rate, not a static
// guess, so it tightens up as the job actually runs.
function estimateEtaSeconds(
  progress: RenderProgress,
  baseline: { time: number; frameIndex: number }
): number | null {
  const elapsedSec = (Date.now() - baseline.time) / 1000;
  const framesDone = progress.frameIndex - baseline.frameIndex;
  if (framesDone <= 0 || elapsedSec <= 0) return null;
  const remaining = progress.totalFrames - progress.frameIndex;
  if (remaining <= 0) return 0;
  return (remaining / framesDone) * elapsedSec;
}

export default function App() {
  const [view, setView] = useState<View>("idle");
  const [overlay, setOverlay] = useState<Overlay>("none");
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [step, setStep] = useState(-1);
  const [error, setError] = useState<string | null>(null);
  const [outputInfo, setOutputInfo] = useState<OutputInfo | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [notify, setNotify] = useState(true);
  const [davinciSafe, setDavinciSafe] = useState(true);
  const [checking, setChecking] = useState(false);
  const [alreadyProcessed, setAlreadyProcessed] = useState(false);
  const [renderProgress, setRenderProgress] = useState<RenderProgress | null>(null);
  const [renderEtaSeconds, setRenderEtaSeconds] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);
  const renderBaselineRef = useRef<{ time: number; frameIndex: number } | null>(null);

  function reset() {
    setView("idle");
    setFile(null);
    setStep(-1);
    setError(null);
    setOutputInfo(null);
    setAlreadyProcessed(false);
    setRenderProgress(null);
    setRenderEtaSeconds(null);
    renderBaselineRef.current = null;
  }

  async function pickFile(f: File | null | undefined) {
    if (!f) return;
    if (f.size > MAX_BYTES) {
      setError(`"${f.name}" is ${formatSize(f.size)} — max is 100MB.`);
      return;
    }

    setError(null);
    let duration: number;
    try {
      duration = await getVideoDuration(f);
    } catch {
      setError(`Couldn't read "${f.name}" — try a different file.`);
      return;
    }
    if (duration > MAX_DURATION_SECONDS) {
      setError(`"${f.name}" is ${formatDuration(duration)} long — max is 1 minute.`);
      return;
    }

    setFile(f);
    setView("configuring");
  }

  async function startProcessing() {
    if (!file) return;

    if (!isSupabaseConfigured) {
      setError(
        "Supabase isn't configured yet — add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in the Vercel project settings, then redeploy."
      );
      setView("error");
      return;
    }

    setError(null);
    setChecking(true);

    // Deterministic on file content + the DaVinci setting (which changes the
    // output itself, via normalization) rather than a random UUID, so the
    // same video processed the same way always lands at the same output
    // location -- that's what lets us detect "already processed" below
    // instead of silently re-running the job every time.
    const baseName = file.name.replace(/\.[^.]+$/, "");
    let contentHash: string;
    try {
      contentHash = await sha256Hex(file);
    } catch (e) {
      setChecking(false);
      setError(e instanceof Error ? e.message : "Couldn't read the file.");
      setView("error");
      return;
    }
    const videoKey = `input/${contentHash}/${sanitizeFileName(file.name)}`;
    const outputPrefix = `sequence_${contentHash}_${davinciSafe ? "dvsafe" : "raw"}`;

    try {
      // The presence of the "_complete.json" marker -- written by the worker
      // only once every frame has been uploaded -- is what "already
      // processed" actually means. Checking for *any* file here would be
      // wrong: a job that died partway through leaves some frames sitting in
      // the bucket, and treating that as "done" would hand back a broken,
      // incomplete result instead of finishing the job.
      const { data: existing, error: listError } = await supabase.storage
        .from(OUTPUT_BUCKET)
        .list(outputPrefix, { limit: 1, search: "_complete.json" });
      if (listError) throw listError;

      if (existing && existing.length > 0) {
        setOutputInfo(await outputInfoForJob(outputPrefix, baseName));
        setAlreadyProcessed(true);
        setView("done");
        return;
      }

      setAlreadyProcessed(false);
      setView("processing");
      setStep(0);

      // Same content hash -> same input key, so skip re-uploading a video
      // that's already sitting in the input bucket from a prior attempt
      // (whether that attempt finished or died partway through).
      const { data: existingInput, error: inputListError } = await supabase.storage
        .from(INPUT_BUCKET)
        .list(`input/${contentHash}`, { limit: 1 });
      if (inputListError) throw inputListError;

      if (!existingInput || existingInput.length === 0) {
        const { error: uploadError } = await supabase.storage
          .from(INPUT_BUCKET)
          .upload(videoKey, file, { upsert: true });
        if (uploadError) throw uploadError;
      }

      setStep(1);
      const { id: jobId } = await startJob({
        input_bucket: INPUT_BUCKET,
        video_key: videoKey,
        output_bucket: OUTPUT_BUCKET,
        output_prefix: outputPrefix,
        davinci_safe: davinciSafe,
      });

      await pollJobUntilDone(jobId, (status: JobStatus, output: unknown) => {
        if (status !== "IN_PROGRESS") return;
        setStep(2);
        const progress = parseRenderProgress(output);
        setRenderProgress(progress);
        if (!progress) return;
        if (!renderBaselineRef.current) {
          renderBaselineRef.current = { time: Date.now(), frameIndex: progress.frameIndex };
          setRenderEtaSeconds(null);
        } else {
          setRenderEtaSeconds(estimateEtaSeconds(progress, renderBaselineRef.current));
        }
      });

      setOutputInfo(await outputInfoForJob(outputPrefix, baseName));
      setView("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      setView("error");
    } finally {
      setChecking(false);
    }
  }

  async function handleDownload() {
    if (!outputInfo) return;
    setDownloading(true);
    setDownloadProgress(0);
    try {
      await downloadOutputAsZip(OUTPUT_BUCKET, outputInfo.prefix, `${outputInfo.baseName}_depth`, setDownloadProgress);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed.");
    } finally {
      setDownloading(false);
    }
  }

  const showFlow = overlay === "none";
  const isDropTarget = showFlow && view === "idle";

  return (
    <div style={styles.page}>
      <Sidebar
        activeOverlay={overlay}
        onUpload={() => setOverlay("none")}
        onAbout={() => setOverlay(overlay === "about" ? "none" : "about")}
        onLogin={() => setOverlay(overlay === "login" ? "none" : "login")}
        onSettings={() => setOverlay(overlay === "settings" ? "none" : "settings")}
      />

      <main
        style={{ ...styles.main, background: dragOver && isDropTarget ? "#e9e9e9" : "#ffffff" }}
        onDragEnter={(e) => {
          e.preventDefault();
          if (!isDropTarget) return;
          dragCounter.current++;
          setDragOver(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          e.preventDefault();
          if (!isDropTarget) return;
          dragCounter.current = Math.max(0, dragCounter.current - 1);
          if (dragCounter.current === 0) setDragOver(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          dragCounter.current = 0;
          setDragOver(false);
          if (!isDropTarget) return;
          pickFile(e.dataTransfer.files?.[0]);
        }}
      >
        {!showFlow && (view === "done" || view === "error") && (
          <div style={{ ...styles.sessionBanner, ...(view === "error" ? styles.sessionBannerError : {}) }}>
            <span>
              {view === "done"
                ? "Your file is ready."
                : "Something went wrong processing your file."}
            </span>
            <button onClick={() => setOverlay("none")} style={styles.sessionBannerLink}>
              {view === "done" ? "Go back to download it" : "Go back for details"}
            </button>
          </div>
        )}

        {showFlow && (
          <>
            {view === "idle" && (
              <IdleView
                dragOver={dragOver}
                error={error}
                fileInputRef={fileInputRef}
                onBrowseClick={() => fileInputRef.current?.click()}
                onFileChange={(e) => pickFile(e.target.files?.[0])}
              />
            )}

            {view === "configuring" && file && (
              <ConfiguringView
                file={file}
                davinciSafe={davinciSafe}
                onSetDavinciSafe={setDavinciSafe}
                onReset={reset}
                onStart={startProcessing}
                checking={checking}
              />
            )}

            {view === "processing" && (
              <ProcessingView step={step} renderProgress={renderProgress} renderEtaSeconds={renderEtaSeconds} />
            )}

            {view === "done" && outputInfo && (
              <DoneView
                outputInfo={outputInfo}
                downloading={downloading}
                downloadProgress={downloadProgress}
                alreadyProcessed={alreadyProcessed}
                onReset={reset}
                onDownload={handleDownload}
              />
            )}

            {view === "error" && <ErrorView message={error ?? "Unknown error"} onReset={reset} />}
          </>
        )}

        {overlay === "about" && <AboutPanel />}
        {overlay === "login" && <LoginPanel />}
        {overlay === "settings" && <SettingsPanel notify={notify} onToggleNotify={() => setNotify((n) => !n)} />}
      </main>
    </div>
  );
}

// ---------- Sidebar ----------

function Sidebar({
  activeOverlay,
  onUpload,
  onAbout,
  onLogin,
  onSettings,
}: {
  activeOverlay: Overlay;
  onUpload: () => void;
  onAbout: () => void;
  onLogin: () => void;
  onSettings: () => void;
}) {
  const items: Array<{ n: string; label: string; bg: string; onClick: () => void; active: boolean; icon: React.ReactNode }> = [
    { n: "01", label: "Upload", bg: "#161616", onClick: onUpload, active: activeOverlay === "none", icon: <UploadIcon stroke="#ffffff" /> },
    { n: "02", label: "About", bg: "#3a3a3a", onClick: onAbout, active: activeOverlay === "about", icon: <PricingIcon /> },
    { n: "03", label: "Sign in", bg: "#5c5c5c", onClick: onLogin, active: activeOverlay === "login", icon: <LoginIcon /> },
    { n: "04", label: "Settings", bg: "#8a8a8a", onClick: onSettings, active: activeOverlay === "settings", icon: <GearIcon /> },
  ];

  return (
    <aside style={styles.sidebar}>
      <div style={styles.logoSwatch} />
      <div style={styles.navList}>
        {items.map((it) => (
          <button
            key={it.n}
            onClick={it.onClick}
            style={{
              ...styles.navCard,
              background: it.bg,
              outline: it.active ? "2px solid #ffffff" : "none",
              outlineOffset: -2,
            }}
          >
            <div style={styles.navCardTop}>
              <span style={styles.navNumber}>{it.n}</span>
              {it.icon}
            </div>
            <span style={styles.navLabel}>{it.label}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}

// ---------- Flow states ----------

function IdleView({
  dragOver,
  error,
  fileInputRef,
  onBrowseClick,
  onFileChange,
}: {
  dragOver: boolean;
  error: string | null;
  fileInputRef: React.RefObject<HTMLInputElement>;
  onBrowseClick: () => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div style={styles.centeredCol}>
      <h1 style={styles.h1}>EZ DEPTH</h1>
      <p style={styles.leadText}>
        One click away from professional depth maps.
      </p>
      <div
        onClick={onBrowseClick}
        style={{
          ...styles.dropzone,
          background: dragOver ? "#e9e9e9" : "#ffffff",
        }}
      >
        <div style={styles.dropIconBox}>
          <UploadIcon />
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 500 }}>Drag a file here or click to browse</div>
          <div style={{ fontSize: 10, color: "#999999", marginTop: 4 }}>
            Outputs as {OUTPUT_FORMAT_LABEL} — up to 1 minute, 6GB
          </div>
          {error && <div style={styles.inlineError}>{error}</div>}
        </div>
        <input ref={fileInputRef} type="file" accept="video/*" onChange={onFileChange} style={{ display: "none" }} />
      </div>
    </div>
  );
}

function ConfiguringView({
  file,
  davinciSafe,
  onSetDavinciSafe,
  onReset,
  onStart,
  checking,
}: {
  file: File;
  davinciSafe: boolean;
  onSetDavinciSafe: (value: boolean) => void;
  onReset: () => void;
  onStart: () => void;
  checking: boolean;
}) {
  return (
    <div style={styles.card}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={styles.fileIconBox}>
          <FileIcon />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={styles.fileName}>{file.name}</div>
          <div style={{ fontSize: 10, color: "#999999" }}>{formatSize(file.size)}</div>
        </div>
        <button onClick={onReset} style={styles.smallIconButton}>
          <CloseIcon size={12} />
        </button>
      </div>

      <div>
        <div style={styles.eyebrow}>Output format</div>
        <div style={styles.radioRow}>
          <RadioDot />
          <span style={{ fontSize: 13 }}>{OUTPUT_FORMAT_LABEL}</span>
        </div>
      </div>

      <div>
        <div style={styles.eyebrow}>Is this used for DaVinci?</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <button
            onClick={() => onSetDavinciSafe(true)}
            style={{
              ...styles.radioRow,
              ...styles.radioRowNarrow,
              cursor: "pointer",
              borderColor: davinciSafe ? "#111111" : "#e5e5e5",
              background: davinciSafe ? "#fafafa" : "#ffffff",
            }}
          >
            <RadioDot filled={davinciSafe} />
            <span style={{ fontSize: 13 }}>Yes</span>
          </button>
          <button
            onClick={() => onSetDavinciSafe(false)}
            style={{
              ...styles.radioRow,
              ...styles.radioRowNarrow,
              cursor: "pointer",
              borderColor: davinciSafe ? "#e5e5e5" : "#111111",
              background: davinciSafe ? "#ffffff" : "#fafafa",
            }}
          >
            <RadioDot filled={!davinciSafe} />
            <span style={{ fontSize: 13 }}>No</span>
          </button>
        </div>
      </div>

      <button onClick={onStart} disabled={checking} style={styles.primaryButton}>
        {checking ? "Checking…" : "Process file"}
      </button>
    </div>
  );
}

function ProcessingView({
  step,
  renderProgress,
  renderEtaSeconds,
}: {
  step: number;
  renderProgress: RenderProgress | null;
  renderEtaSeconds: number | null;
}) {
  const RENDER_STEP_INDEX = 2; // "Preparing output" -- the RunPod render itself
  const total = STEP_LABELS.length;
  // Real per-frame progress from the worker replaces the flat "half credit for
  // being active" guess once we have it -- the bar actually tracks the render
  // instead of just sitting at a fixed spot for however long that step takes.
  const renderFraction =
    renderProgress && renderProgress.totalFrames > 0
      ? renderProgress.frameIndex / renderProgress.totalFrames
      : 0.5;
  const completedUnits = STEP_LABELS.reduce(
    (acc, _label, i) =>
      acc + (step > i ? 1 : step === i ? (i === RENDER_STEP_INDEX ? renderFraction : 0.5) : 0),
    0
  );
  const percent = Math.min(100, Math.round((completedUnits / total) * 100));

  return (
    <div style={{ ...styles.card, padding: "32px 24px", gap: 24 }}>
      <div>
        <div style={styles.progressTrack}>
          <div style={{ ...styles.progressFill, width: `${percent}%` }} />
        </div>
        <div style={styles.progressLabel}>{percent}%</div>
      </div>
      {STEP_LABELS.map((label, i) => {
        const done = step > i;
        const active = step === i;
        const ringColor = done || active ? "#111111" : "#e0e0e0";
        const fillColor = done ? "#111111" : "#ffffff";
        const textColor = done || active ? "#111111" : "#aaaaaa";
        const showRenderDetail = active && i === RENDER_STEP_INDEX && renderProgress;
        return (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {active ? (
              <span className="step-spinner" />
            ) : (
              <span style={{ ...styles.stepRing, borderColor: ringColor, background: fillColor }}>
                {done && <CheckIcon />}
              </span>
            )}
            <span style={{ fontSize: 13, color: textColor }}>
              {label}
              {showRenderDetail && (
                <span style={{ color: "#999999", fontWeight: 400 }}>
                  {" "}
                  — {renderProgress!.frameIndex}/{renderProgress!.totalFrames} frames
                  {renderEtaSeconds != null && renderEtaSeconds > 0 && (
                    <> · ~{formatDuration(Math.ceil(renderEtaSeconds))} left</>
                  )}
                </span>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function DoneView({
  outputInfo,
  downloading,
  downloadProgress,
  alreadyProcessed,
  onReset,
  onDownload,
}: {
  outputInfo: OutputInfo;
  downloading: boolean;
  downloadProgress: number;
  alreadyProcessed: boolean;
  onReset: () => void;
  onDownload: () => void;
}) {
  const downloadPercent = Math.round(downloadProgress * 100);
  return (
    <div style={{ ...styles.card, alignItems: "center", padding: "36px 24px" }}>
      <div style={styles.doneCheckCircle}>
        <CheckIcon size={18} stroke="#ffffff" width={2.5} />
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 15, fontWeight: 600 }}>
          {alreadyProcessed ? "Already processed — ready to download" : "Your file is ready"}
        </div>
        {alreadyProcessed && (
          <div style={{ fontSize: 11, color: "#999999", marginTop: 2 }}>
            This exact file (with the same DaVinci setting) was processed before, so we
            skipped straight to your existing result.
          </div>
        )}
        <div style={{ fontSize: 13, color: "#666666", marginTop: 4 }}>
          {outputInfo.baseName}_depth.zip · {outputInfo.frameCount} frames (matches job output)
        </div>
      </div>
      {downloading && (
        <div style={{ width: "100%", maxWidth: 320 }}>
          <div style={styles.progressTrack}>
            <div style={{ ...styles.progressFill, width: `${downloadPercent}%` }} />
          </div>
          <div style={styles.progressLabel}>
            {downloadPercent < 90 ? "Downloading frames…" : "Packaging zip…"} {downloadPercent}%
          </div>
        </div>
      )}
      <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
        <button onClick={onReset} style={styles.secondaryButton}>
          Start over
        </button>
        <button onClick={onDownload} disabled={downloading} style={styles.primaryButtonSmall}>
          {downloading ? `Zipping… ${downloadPercent}%` : "Download"}
        </button>
      </div>
    </div>
  );
}

function ErrorView({ message, onReset }: { message: string; onReset: () => void }) {
  return (
    <div style={{ ...styles.card, alignItems: "center", padding: "36px 24px" }}>
      <div style={{ ...styles.doneCheckCircle, background: "#b3261e" }}>
        <CloseIcon size={16} stroke="#ffffff" />
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 15, fontWeight: 600 }}>Processing failed</div>
        <div style={{ fontSize: 13, color: "#666666", marginTop: 4, maxWidth: 420 }}>{message}</div>
      </div>
      <button onClick={onReset} style={{ ...styles.secondaryButton, marginTop: 6 }}>
        Try again
      </button>
    </div>
  );
}

// ---------- Overlays ----------

function PanelHeader({ title }: { title: string }) {
  return (
    <div style={styles.panelHeader}>
      <h2 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>{title}</h2>
    </div>
  );
}

function AboutPanel() {
  const steps = [
    { n: 1, title: "Upload your video", body: "Up to 1min and 6 GB" },
    { n: 2, title: "Download it & Import it", body: "Import the PNG sequence as a matte, drop it into the clip node graph, & link the matte's alpha channel to any node you want." },
    { n: 3, title: "Control your depth map setting with NodeKey.", body: "Adjust the offset parameter in NodeKey (NodeKey output) to control how far or close the effect reaches into the image." },
  ];
  const videos = [
    { title: "Demo", src: "/videos/ez-depth-demo.mp4" },
    { title: "Tutorial", src: "/videos/ez-depth-tutorial.mp4" },
  ];
  const tiers: Array<{ name: string; detail: string; price: string; period?: string; active: boolean }> = [
    { name: "Per Second", detail: "This service is calculated based on second of use", price: "$xx", active: false },
    { name: "Other Option", detail: "Consectetur adipiscing elit", price: "$xx", active: true },
  ];

  return (
    <div style={styles.panelRoot}>
      <PanelHeader title="About" />
      <div style={styles.panelBody}>
        {videos.map((v) => (
          <div key={v.src}>
            <div style={styles.eyebrow}>{v.title}</div>
            <video src={v.src} controls playsInline preload="metadata" style={styles.aboutVideo} />
          </div>
        ))}

        <div>
          <div style={styles.eyebrow}>How it works</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {steps.map((s) => (
              <div key={s.n} style={{ display: "flex", gap: 12 }}>
                <span style={styles.numberBadge}>{s.n}</span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>{s.title}</div>
                  <div style={{ fontSize: 10, color: "#666666", marginTop: 2, lineHeight: 1.5 }}>{s.body}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div style={styles.eyebrow}>Pricing</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {tiers.map((t) => (
              <div
                key={t.name}
                style={{
                  ...styles.tierRow,
                  borderColor: t.active ? "#111111" : "#e5e5e5",
                  background: t.active ? "#fafafa" : "transparent",
                }}
              >
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>{t.name}</div>
                  <div style={{ fontSize: 10, color: "#999999", marginTop: 2 }}>{t.detail}</div>
                </div>
                <div style={{ fontSize: 15, fontWeight: 600 }}>
                  {t.price}
                  {t.period && <span style={{ fontSize: 10, color: "#999999", fontWeight: 400 }}>{t.period}</span>}
                </div>
              </div>
            ))}
          </div>
        </div>

        <p style={{ fontSize: 10, color: "#999999", lineHeight: 1.6, margin: 0 }}>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et
          dolore magna aliqua.
        </p>
      </div>
    </div>
  );
}

function LoginPanel() {
  return (
    <div style={styles.panelRoot}>
      <div style={styles.loginCenter}>
        <div>
          <span style={styles.eyebrow}>Sign in</span>
          <h2 style={{ fontSize: 22, fontWeight: 600, margin: "8px 0 0 0" }}>Welcome back</h2>
          <p style={{ fontSize: 13, color: "#666666", margin: "8px 0 0 0", lineHeight: 1.5 }}>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          </p>
        </div>
        <button style={styles.oauthButton}>
          <span style={styles.gBadge}>G</span>
          Continue with Google
        </button>
        <p style={{ fontSize: 11, color: "#999999", textAlign: "center", lineHeight: 1.5, margin: 0 }}>
          By continuing you agree to our lorem ipsum terms and privacy policy.
        </p>
      </div>
    </div>
  );
}

function SettingsPanel({
  notify,
  onToggleNotify,
}: {
  notify: boolean;
  onToggleNotify: () => void;
}) {
  return (
    <div style={styles.panelRoot}>
      <PanelHeader title="Settings" />
      <div style={styles.panelBody}>
        <div>
          <div style={styles.eyebrow}>Account</div>
          <div style={styles.accountRow}>
            <div style={styles.avatar} />
            <div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Lorem Ipsum</div>
              <div style={{ fontSize: 10, color: "#999999" }}>lorem@ipsum.com</div>
            </div>
          </div>
        </div>

        <div>
          <div style={styles.eyebrow}>Default output</div>
          <div style={styles.radioRow}>
            <RadioDot />
            <span style={{ fontSize: 13 }}>{OUTPUT_FORMAT_LABEL}</span>
          </div>
        </div>

        <div>
          <div style={styles.eyebrow}>Notifications</div>
          <div style={styles.tierRow}>
            <div>
              <div style={{ fontSize: 13 }}>Email when processing finishes</div>
              <div style={{ fontSize: 10, color: "#999999", marginTop: 2 }}>Lorem ipsum dolor sit amet.</div>
            </div>
            <button onClick={onToggleNotify} style={{ ...styles.toggle, background: notify ? "#111111" : "#e0e0e0" }}>
              <span style={{ ...styles.toggleKnob, left: notify ? 18 : 2 }} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- Icons ----------

function UploadIcon({ stroke = "#111111" }: { stroke?: string }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="1.6">
      <path d="M12 16V4M12 4l-5 5M12 4l5 5" />
      <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
    </svg>
  );
}

function PricingIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1v22M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
    </svg>
  );
}

function LoginIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
      <path d="M10 17l5-5-5-5M15 12H3" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#111111" strokeWidth="1.6">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
    </svg>
  );
}

function CloseIcon({ size = 12, stroke = "#666666" }: { size?: number; stroke?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="2">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

function CheckIcon({ size = 11, stroke = "#ffffff", width = 3 }: { size?: number; stroke?: string; width?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={width}>
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function RadioDot({ filled = true }: { filled?: boolean }) {
  return (
    <span style={styles.radioOuter}>
      {filled && <span style={styles.radioInner} />}
    </span>
  );
}

// ---------- Styles ----------

const styles: Record<string, CSSProperties> = {
  page: {
    position: "relative",
    width: "100%",
    minHeight: "100vh",
    background: "#f2f2f7",
    fontFamily: "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",
    color: "#111111",
    display: "flex",
    height: "100vh",
    padding: 14,
    boxSizing: "border-box",
    gap: 14,
  },
  sidebar: {
    width: 98,
    flexShrink: 0,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  logoSwatch: {
    width: 27,
    height: 25,
    background: "#b6b0a7",
    borderRadius: 8,
    marginLeft: 4,
  },
  navList: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  navCard: {
    height: 100,
    flexShrink: 0,
    border: "none",
    borderRadius: 14,
    padding: 14,
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
    cursor: "pointer",
    textAlign: "left",
    width: "100%",
  },
  navCardTop: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
  },
  navNumber: { fontSize: 14, fontWeight: 800, color: "#ffffff" },
  navLabel: { fontSize: 15, fontWeight: 700, color: "#ffffff" },
  main: {
    flex: 1,
    height: "100%",
    boxSizing: "border-box",
    overflow: "hidden",
    position: "relative",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
    background: "#ffffff",
    borderRadius: 20,
  },
  centeredCol: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    textAlign: "center",
    gap: 8,
  },
  h1: { fontSize: 34, fontWeight: 600, margin: 0, letterSpacing: "-0.01em" },
  leadText: { fontSize: 15, color: "#666666", margin: "0 0 28px 0", maxWidth: 440, lineHeight: 1.5 },
  dropzone: {
    width: 599,
    maxWidth: "100%",
    border: "1.5px dashed #d8d8d8",
    borderRadius: 10,
    padding: "56px 24px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 14,
    cursor: "pointer",
    transition: "border-color .15s, background .15s",
    boxSizing: "border-box",
  },
  dropIconBox: {
    width: 44,
    height: 44,
    border: "1px solid #d8d8d8",
    borderRadius: 8,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#ffffff",
  },
  inlineError: { fontSize: 12, color: "#b3261e", marginTop: 8 },
  sessionBanner: {
    position: "absolute",
    top: 16,
    left: 24,
    right: 24,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexWrap: "wrap",
    gap: 8,
    padding: "10px 16px",
    borderRadius: 8,
    background: "#f0f7f0",
    border: "1px solid #b8dab8",
    color: "#1a5c1a",
    fontSize: 13,
    textAlign: "center",
  },
  sessionBannerError: {
    background: "#fbebea",
    border: "1px solid #e8a6a1",
    color: "#8a2c25",
  },
  sessionBannerLink: {
    background: "none",
    border: "none",
    padding: 0,
    font: "inherit",
    fontWeight: 700,
    color: "inherit",
    textDecoration: "underline",
    cursor: "pointer",
  },
  card: {
    width: 600,
    maxWidth: "100%",
    minHeight: 196,
    border: "1px solid #e5e5e5",
    borderRadius: 10,
    padding: 20,
    textAlign: "left",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    gap: 20,
    boxSizing: "border-box",
  },
  fileIconBox: {
    width: 36,
    height: 36,
    border: "1px solid #e5e5e5",
    borderRadius: 6,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  fileName: { fontSize: 13, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  smallIconButton: {
    width: 26,
    height: 26,
    border: "1px solid #e5e5e5",
    borderRadius: 6,
    background: "#ffffff",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  eyebrow: {
    fontSize: 10,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "#888888",
    marginBottom: 10,
  },
  radioRow: {
    display: "flex",
    alignItems: "center",
    gap: 9,
    border: "1px solid #111111",
    borderRadius: 7,
    padding: "10px 12px",
    background: "#fafafa",
    font: "inherit",
    color: "#111111",
    textAlign: "left",
  },
  radioRowNarrow: {
    width: 245,
  },
  radioOuter: {
    width: 15,
    height: 15,
    borderRadius: "50%",
    border: "1.5px solid #111111",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  radioInner: { width: 7, height: 7, borderRadius: "50%", background: "#111111" },
  primaryButton: {
    height: 42,
    background: "#111111",
    color: "#ffffff",
    border: "none",
    borderRadius: 7,
    fontSize: 14,
    cursor: "pointer",
  },
  primaryButtonSmall: {
    height: 38,
    padding: "0 18px",
    background: "#111111",
    color: "#ffffff",
    border: "none",
    borderRadius: 7,
    fontSize: 13,
    cursor: "pointer",
  },
  secondaryButton: {
    height: 38,
    padding: "0 16px",
    background: "#ffffff",
    color: "#111111",
    border: "1px solid #e5e5e5",
    borderRadius: 7,
    fontSize: 13,
    cursor: "pointer",
  },
  progressTrack: {
    width: "100%",
    height: 6,
    borderRadius: 3,
    background: "#eeeeee",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    background: "#111111",
    borderRadius: 3,
    transition: "width .3s ease",
  },
  progressLabel: {
    fontSize: 11,
    color: "#999999",
    marginTop: 6,
    textAlign: "right",
  },
  stepRing: {
    width: 20,
    height: 20,
    borderRadius: "50%",
    border: "1.5px solid",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  doneCheckCircle: {
    width: 40,
    height: 40,
    borderRadius: "50%",
    background: "#111111",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  panelRoot: {
    width: "100%",
    height: "100%",
    display: "flex",
    flexDirection: "column",
    textAlign: "left",
  },
  panelHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 20,
    flexShrink: 0,
  },
  panelBody: {
    display: "flex",
    flexDirection: "column",
    gap: 24,
    overflowY: "auto",
  },
  aboutVideo: {
    display: "block",
    width: "100%",
    borderRadius: 8,
    background: "#000000",
  },
  numberBadge: {
    width: 22,
    height: 22,
    borderRadius: "50%",
    border: "1px solid #111111",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 10,
    flexShrink: 0,
  },
  tierRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    border: "1px solid #e5e5e5",
    borderRadius: 8,
    padding: 14,
  },
  loginCenter: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    gap: 22,
    maxWidth: 300,
    margin: "0 auto",
    width: "100%",
    textAlign: "center",
  },
  oauthButton: {
    height: 44,
    width: "100%",
    border: "1px solid #d8d8d8",
    borderRadius: 7,
    background: "#ffffff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    cursor: "pointer",
    fontSize: 13,
  },
  gBadge: {
    width: 17,
    height: 17,
    borderRadius: "50%",
    border: "1.5px solid #111111",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 10,
    fontWeight: 700,
  },
  accountRow: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    border: "1px solid #e5e5e5",
    borderRadius: 8,
    padding: 14,
  },
  avatar: { width: 36, height: 36, borderRadius: "50%", background: "#111111" },
  toggle: {
    width: 38,
    height: 22,
    borderRadius: 11,
    border: "none",
    position: "relative",
    cursor: "pointer",
    flexShrink: 0,
  },
  toggleKnob: {
    position: "absolute",
    top: 2,
    width: 18,
    height: 18,
    borderRadius: "50%",
    background: "#ffffff",
    transition: "left .15s",
  },
};
