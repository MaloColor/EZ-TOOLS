import { useRef, useState, type CSSProperties } from "react";
import { supabase, isSupabaseConfigured, INPUT_BUCKET, OUTPUT_BUCKET } from "./lib/supabaseClient";
import { startJob, pollJobUntilDone, type JobStatus } from "./lib/job";
import { downloadOutputAsZip } from "./lib/downloadZip";

type View = "idle" | "configuring" | "processing" | "done" | "error";
type Overlay = "none" | "about" | "login" | "settings";

const MAX_BYTES = 25 * 1024 * 1024;
const OUTPUT_FORMAT_LABEL = "EXR Depth Sequence";
const STEP_LABELS = ["Uploading", "Analyzing", "Preparing output"];

function formatSize(bytes: number): string {
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(0)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

function sanitizeFileName(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]/g, "_");
}

interface OutputInfo {
  prefix: string;
  baseName: string;
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
  const [notify, setNotify] = useState(true);
  const [davinciSafe, setDavinciSafe] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function reset() {
    setView("idle");
    setFile(null);
    setStep(-1);
    setError(null);
    setOutputInfo(null);
  }

  function pickFile(f: File | null | undefined) {
    if (!f) return;
    if (f.size > MAX_BYTES) {
      setError(`"${f.name}" is ${formatSize(f.size)} — max is 25MB.`);
      return;
    }
    setError(null);
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

    setView("processing");
    setStep(0);
    setError(null);

    const jobUuid = crypto.randomUUID();
    const videoKey = `${jobUuid}/${sanitizeFileName(file.name)}`;
    const outputPrefix = `sequence_${jobUuid}`;

    try {
      const { error: uploadError } = await supabase.storage
        .from(INPUT_BUCKET)
        .upload(videoKey, file);
      if (uploadError) throw uploadError;

      setStep(1);
      const { id: jobId } = await startJob({
        input_bucket: INPUT_BUCKET,
        video_key: videoKey,
        output_bucket: OUTPUT_BUCKET,
        output_prefix: outputPrefix,
        davinci_safe: davinciSafe,
      });

      await pollJobUntilDone(jobId, (status: JobStatus) => {
        if (status === "IN_PROGRESS") setStep(2);
      });

      setOutputInfo({
        prefix: outputPrefix,
        baseName: file.name.replace(/\.[^.]+$/, ""),
      });
      setView("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      setView("error");
    }
  }

  async function handleDownload() {
    if (!outputInfo) return;
    setDownloading(true);
    try {
      await downloadOutputAsZip(OUTPUT_BUCKET, outputInfo.prefix, `${outputInfo.baseName}_depth`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed.");
    } finally {
      setDownloading(false);
    }
  }

  const showFlow = overlay === "none";

  return (
    <div style={styles.page}>
      <Sidebar
        activeOverlay={overlay}
        onUpload={() => {
          setOverlay("none");
          reset();
        }}
        onAbout={() => setOverlay(overlay === "about" ? "none" : "about")}
        onLogin={() => setOverlay(overlay === "login" ? "none" : "login")}
        onSettings={() => setOverlay(overlay === "settings" ? "none" : "settings")}
      />

      <main style={styles.main}>
        {showFlow && (
          <>
            {view === "idle" && (
              <IdleView
                dragOver={dragOver}
                error={error}
                fileInputRef={fileInputRef}
                onBrowseClick={() => fileInputRef.current?.click()}
                onFileChange={(e) => pickFile(e.target.files?.[0])}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  pickFile(e.dataTransfer.files?.[0]);
                }}
              />
            )}

            {view === "configuring" && file && (
              <ConfiguringView
                file={file}
                davinciSafe={davinciSafe}
                onSetDavinciSafe={setDavinciSafe}
                onReset={reset}
                onStart={startProcessing}
              />
            )}

            {view === "processing" && <ProcessingView step={step} />}

            {view === "done" && outputInfo && (
              <DoneView
                outputInfo={outputInfo}
                downloading={downloading}
                onReset={reset}
                onDownload={handleDownload}
              />
            )}

            {view === "error" && <ErrorView message={error ?? "Unknown error"} onReset={reset} />}
          </>
        )}

        {overlay === "about" && <AboutPanel onClose={() => setOverlay("none")} />}
        {overlay === "login" && <LoginPanel onClose={() => setOverlay("none")} />}
        {overlay === "settings" && (
          <SettingsPanel notify={notify} onToggleNotify={() => setNotify((n) => !n)} onClose={() => setOverlay("none")} />
        )}
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
  const items: Array<{ n: string; label: string; bg: string; onClick: () => void; active: boolean }> = [
    { n: "01", label: "Upload", bg: "#161616", onClick: onUpload, active: activeOverlay === "none" },
    { n: "02", label: "About", bg: "#3a3a3a", onClick: onAbout, active: activeOverlay === "about" },
    { n: "03", label: "Sign in", bg: "#5c5c5c", onClick: onLogin, active: activeOverlay === "login" },
    { n: "04", label: "Settings", bg: "#8a8a8a", onClick: onSettings, active: activeOverlay === "settings" },
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
              <ArrowIcon />
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
  onDragOver,
  onDragLeave,
  onDrop,
}: {
  dragOver: boolean;
  error: string | null;
  fileInputRef: React.RefObject<HTMLInputElement>;
  onBrowseClick: () => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
}) {
  return (
    <div style={styles.centeredCol}>
      <h1 style={styles.h1}>Lorem ipsum dolor sit amet</h1>
      <p style={styles.leadText}>
        Consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
      </p>
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={onBrowseClick}
        style={{
          ...styles.dropzone,
          borderColor: dragOver ? "#111111" : "#d8d8d8",
          background: dragOver ? "#fafafa" : "#ffffff",
        }}
      >
        <div style={styles.dropIconBox}>
          <UploadIcon />
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 500 }}>Drag a file here or click to browse</div>
          <div style={{ fontSize: 10, color: "#999999", marginTop: 4 }}>
            Outputs as {OUTPUT_FORMAT_LABEL} — up to 25MB
          </div>
        </div>
        <input ref={fileInputRef} type="file" accept="video/*" onChange={onFileChange} style={{ display: "none" }} />
      </div>
      {error && <div style={styles.inlineError}>{error}</div>}
    </div>
  );
}

function ConfiguringView({
  file,
  davinciSafe,
  onSetDavinciSafe,
  onReset,
  onStart,
}: {
  file: File;
  davinciSafe: boolean;
  onSetDavinciSafe: (value: boolean) => void;
  onReset: () => void;
  onStart: () => void;
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

      <button onClick={onStart} style={styles.primaryButton}>
        Process file
      </button>
    </div>
  );
}

function ProcessingView({ step }: { step: number }) {
  return (
    <div style={{ ...styles.card, padding: "32px 24px" }}>
      {STEP_LABELS.map((label, i) => {
        const done = step > i;
        const active = step === i;
        const ringColor = done || active ? "#111111" : "#e0e0e0";
        const fillColor = done ? "#111111" : "#ffffff";
        const textColor = done || active ? "#111111" : "#aaaaaa";
        return (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ ...styles.stepRing, borderColor: ringColor, background: fillColor }}>
              {done && <CheckIcon />}
              {active && <span style={styles.spinnerDot} />}
            </span>
            <span style={{ fontSize: 13, color: textColor }}>{label}</span>
          </div>
        );
      })}
    </div>
  );
}

function DoneView({
  outputInfo,
  downloading,
  onReset,
  onDownload,
}: {
  outputInfo: OutputInfo;
  downloading: boolean;
  onReset: () => void;
  onDownload: () => void;
}) {
  return (
    <div style={{ ...styles.card, alignItems: "center", padding: "36px 24px" }}>
      <div style={styles.doneCheckCircle}>
        <CheckIcon size={18} stroke="#ffffff" width={2.5} />
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 15, fontWeight: 600 }}>Your file is ready</div>
        <div style={{ fontSize: 13, color: "#666666", marginTop: 4 }}>
          {outputInfo.baseName}_depth.zip
        </div>
      </div>
      <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
        <button onClick={onReset} style={styles.secondaryButton}>
          Start over
        </button>
        <button onClick={onDownload} disabled={downloading} style={styles.primaryButtonSmall}>
          {downloading ? "Zipping…" : "Download"}
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

function PanelHeader({ title, onClose }: { title: string; onClose: () => void }) {
  return (
    <div style={styles.panelHeader}>
      <h2 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>{title}</h2>
      <button onClick={onClose} style={styles.smallIconButton}>
        <CloseIcon size={14} />
      </button>
    </div>
  );
}

function AboutPanel({ onClose }: { onClose: () => void }) {
  const steps = [
    { n: 1, title: "Lorem ipsum upload", body: "Consectetur adipiscing elit, sed do eiusmod tempor incididunt." },
    { n: 2, title: "Dolore magna processing", body: "Ut enim ad minim veniam, quis nostrud exercitation ullamco." },
    { n: 3, title: "Laboris nisi download", body: "Duis aute irure dolor in reprehenderit in voluptate velit." },
  ];
  const tiers: Array<{ name: string; detail: string; price: string; period?: string; active: boolean }> = [
    { name: "Per Second", detail: "This service is calculated based on second of use", price: "$xx", active: false },
    { name: "Other Option", detail: "Consectetur adipiscing elit", price: "$xx", active: true },
  ];

  return (
    <div style={styles.panelRoot}>
      <PanelHeader title="About" onClose={onClose} />
      <div style={styles.panelBody}>
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

function LoginPanel({ onClose }: { onClose: () => void }) {
  return (
    <div style={styles.panelRoot}>
      <div style={{ display: "flex", justifyContent: "flex-end", flexShrink: 0 }}>
        <button onClick={onClose} style={styles.smallIconButton}>
          <CloseIcon size={14} />
        </button>
      </div>
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
  onClose,
}: {
  notify: boolean;
  onToggleNotify: () => void;
  onClose: () => void;
}) {
  return (
    <div style={styles.panelRoot}>
      <PanelHeader title="Settings" onClose={onClose} />
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

function ArrowIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2">
      <path d="M7 17 17 7M7 7h10v10" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#111111" strokeWidth="1.6">
      <path d="M12 16V4M12 4l-5 5M12 4l5 5" />
      <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
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
  navNumber: { fontSize: 10, fontWeight: 800, color: "#ffffff" },
  navLabel: { fontSize: 10, fontWeight: 700, color: "#ffffff" },
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
  inlineError: { fontSize: 12, color: "#b3261e", marginTop: 12 },
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
  spinnerDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#111111",
    animation: "spin 1s linear infinite",
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
