# Setup & deploy

This is a Vite + React + TypeScript app with two Vercel serverless functions
in `/api`. It implements the "File Drop" design (idle → configure output →
process → done, plus About / Sign in / Settings side panels) and wires the
upload/process/download flow to a real Supabase Storage + RunPod backend —
specifically the `video-depth-worker` in your `EZ-TOOLS` repo.

Marketing copy (headline, About steps, pricing tiers, sign-in text) is left
as Lorem Ipsum from the original design, per your call — swap it in
`src/App.tsx` whenever you have real copy.

## 1. Supabase

Create **one** Storage bucket (name must match, or update
`src/lib/supabaseClient.ts`):

- `depth-outputs` — holds both the uploaded input video (under `input/...`)
  and the worker's output EXR sequence (under `sequence_*/...`), kept apart
  by path prefix rather than by bucket. There's no per-user auth yet to make
  bucket-level separation meaningful, so one bucket is simpler to administer
  — split it back into two (e.g. to set different retention/cleanup rules,
  or to make raw uploads private once auth exists) if that changes.

The browser uploads/downloads using the **anon** key, so it needs **both** an
insert policy and a select policy — missing either one breaks a different
half of the flow (uploads silently work but nothing can be read back, or vice
versa). Minimal example (tighten as needed — e.g. scope to authenticated
users once real auth is wired up instead of the design's placeholder
"Sign in" button):

```sql
create policy "anon insert" on storage.objects
  for insert to anon
  with check (bucket_id = 'depth-outputs');

create policy "anon select" on storage.objects
  for select to anon
  using (bucket_id = 'depth-outputs');
```

Sanity-check both directions before assuming it's wired up — upload a test
object with the anon key, then immediately try to read it back with the same
key. A 200 on upload does **not** imply the select policy exists; an object
that "uploads fine" can still 404 on every subsequent read if only the
insert policy was created.

The worker itself (running on RunPod) uses the **service role** key
server-side, per the existing `handler.py` — that key bypasses RLS entirely
and never touches this front end.

## 2. RunPod

Build/deploy `video-depth-worker`'s Dockerfile as a RunPod Serverless
endpoint. In the endpoint's environment variables, set the same
`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` the worker expects. Note the
endpoint ID — you'll need it below.

## 3. Environment variables

Copy `.env.example` to `.env.local` for local dev, and set the same keys in
Vercel (Project Settings → Environment Variables):

| Variable | Where it's used | Exposed to browser? |
|---|---|---|
| `VITE_SUPABASE_URL` | client | yes |
| `VITE_SUPABASE_ANON_KEY` | client | yes |
| `RUNPOD_API_KEY` | `/api/*` only | no |
| `RUNPOD_ENDPOINT_ID` | `/api/*` only | no |

`RUNPOD_API_KEY` is never sent to the browser — `api/start-job.ts` and
`api/job-status.ts` are the only things that read it, which is why the app
needs those two serverless functions instead of calling RunPod directly from
the client.

## 4. Local dev

```bash
npm install
npx vercel dev   # runs both the Vite front end and the /api functions
```

(`npm run dev` alone only serves the front end — the /api routes need
`vercel dev`, or `vercel link` + `vercel env pull` first if you haven't
linked the project yet.)

## 5. Deploy

```bash
npx vercel        # first deploy / preview, links the project if needed
npx vercel --prod # production deploy
```

Vercel auto-detects the Vite framework and turns `/api/*.ts` into serverless
functions — no `vercel.json` needed.

## Known simplifications

- The worker reports coarse RunPod job status (`IN_QUEUE` / `IN_PROGRESS` /
  `COMPLETED`), not per-frame progress, so the 3-step "Uploading / Analyzing
  / Preparing output" UI is an approximation: step 2 lights up on
  `IN_QUEUE`, step 3 on `IN_PROGRESS`. There's no real signal to distinguish
  "analyzing" from "writing EXRs" within `IN_PROGRESS`.
- The worker uploads one EXR per frame rather than a single output file, so
  "Download" zips every frame client-side with JSZip after fetching each one
  from Supabase. For very long clips this pulls the whole sequence into the
  browser before saving — fine for short clips, but a server-side zip step
  would scale better if that becomes a problem.
- The output-format selector always shows "EXR Depth Sequence" since that's
  the only thing the current worker produces; the design's placeholder radio
  UI is kept in case more formats are added later.
- "Sign in" and the account row in Settings are still the design's static
  mockup — no real auth is wired up yet.
