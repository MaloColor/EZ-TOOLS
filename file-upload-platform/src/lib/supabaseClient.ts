import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

export const isSupabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey);

if (!isSupabaseConfigured) {
  console.error(
    "Missing VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY env vars. See SETUP.md. " +
      "The app will still render, but upload/process will show a config error."
  );
}

// createClient() throws synchronously on a missing/invalid URL, and this
// runs at module load time — an unconfigured deploy would otherwise crash
// before React ever renders, producing a blank page instead of a working UI
// with a clear in-app error. Fall back to a placeholder so the client always
// constructs; real calls are gated behind isSupabaseConfigured.
export const supabase = createClient(
  supabaseUrl || "https://placeholder.supabase.co",
  supabaseAnonKey || "placeholder-anon-key"
);

export const INPUT_BUCKET = "raw-videos";
export const OUTPUT_BUCKET = "depth-outputs";
