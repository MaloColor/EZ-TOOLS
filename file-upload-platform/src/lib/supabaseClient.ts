import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!supabaseUrl || !supabaseAnonKey) {
  // Fails loudly in dev/build rather than silently making requests to "undefined".
  console.error(
    "Missing VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY env vars. See SETUP.md."
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export const INPUT_BUCKET = "raw-videos";
export const OUTPUT_BUCKET = "depth-outputs";
