# Vodafone AI Sales — backend

An AI voice agent ("Sam") rings small Irish businesses on behalf of Vodafone. It says it is an AI, qualifies interest and books a follow-up. Each call's outcome, transcript and AI analysis land in Supabase, and the web app (`vodefoneAI_frontend_v2`) shows them to the sales team.

## How it fits together

```
web app ──reads/writes──▶ Supabase (Postgres + Auth, row-level security)
   │                         ▲
   └─▶ src/api ──LiveKit──┐  │ saves calls, transcripts, analysis
                          ▼  │
src/dialler ──dispatch──▶ LiveKit Cloud ──▶ src/agent (deployed) ──SIP──▶ phone
```

| Folder | What it does | Runs where |
| --- | --- | --- |
| `src/agent` | The voice agent: prompt, speech, voicemail detection, `record_outcome`. Catalogue and objection answers are `services.md` and `objections.md`. | LiveKit Cloud (`lk agent deploy`) |
| `src/dialler` | Every few seconds, rings due leads of running campaigns: calling hours and days, daily cap, retries, callbacks, allowlist. | Your machine (`scripts/demo.sh`) |
| `src/analysis` | After each answered call: score, brief, next step, objections, qualification. | Inside the agent |
| `src/api` | Small FastAPI service holding the LiveKit secret: lists live calls and issues hidden listen-only tokens. | Your machine (`scripts/demo.sh`) |
| `src/shared` | All SQL (`db.py`) and saving a call's outcome (`results.py`). | — |
| `supabase/migrations` | Database schema and access rules. | Supabase (`supabase db push`) |
| `scripts` | `demo.sh` runs the API and dialler; `reset_demo.py` makes test numbers callable again. | Your machine |
| `data` | The Vodafone Ireland business catalogue scraped on 29 Oct 2026. | — |

## Setup

Needs Python 3.13, [uv](https://docs.astral.sh/uv/), the LiveKit CLI (`lk`) and the Supabase CLI.

```bash
uv sync
cp .env.example .env   # then fill it in
```

`.env` holds real keys and the production database URL. It is git-ignored; never commit it.

| Variable | Needed for |
| --- | --- |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Agent, dialler, API |
| `DATABASE_URL` | Everything. Use Supabase's Session pooler string (the direct one is IPv6-only). |
| `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `FRONTEND_ORIGIN` | API |
| `DIAL_ALLOWLIST` | Dialler. Comma-separated numbers it may ring. Required; `demo.sh` refuses `*`. |
| `ELEVEN_API_KEY`, `ELEVEN_VOICE_ID`, `BUSINESS_NAME`, `SIP_OUTBOUND_TRUNK_ID` | Agent (when run locally; the deployed agent uses LiveKit secrets) |

## Running a demo

1. Deploy the agent after any change under `src/agent` or `src/analysis`. Its settings live in LiveKit secrets (`lk agent secrets`; change with `lk agent update-secrets`).
   ```bash
   lk agent deploy
   ```
2. Start the API and dialler. Ctrl-C stops both, and so does either one dying.
   ```bash
   ./scripts/demo.sh
   ```
3. Start the web app, sign in, create a campaign and launch it. The dialler only rings numbers in `DIAL_ALLOWLIST`, inside the campaign's calling hours and days.
4. Between runs, stop `demo.sh` and reset the test numbers:
   ```bash
   uv run python scripts/reset_demo.py
   ```

Talk to the agent without a phone line: `uv run python -m agent.main console`.

## Database changes

Add a new timestamped file to `supabase/migrations`; never edit an applied one. Then:

```bash
supabase db push --db-url "$DATABASE_URL"
```

This changes the shared database: try `--dry-run` first.

## Tests

The tests rebuild the schema in a local Postgres database (`TEST_DATABASE_URL`, default `postgresql:///vodafoneai_test`) and never touch production.

```bash
createdb vodafoneai_test   # first time only
LIVEKIT_API_KEY= LIVEKIT_API_SECRET= LIVEKIT_URL= uv run pytest
```

Blanking the LiveKit keys skips the tests that call real AI models, which spend credits. Without them, two agent tests fail with "api_key is required"; that is expected.

Persona eval (30 scripted customers, slow, spends credits). Run it after any change to the prompt, catalogue or objection answers:

```bash
uv run pytest -m eval
```

## Rules that keep real customers safe

- The dialler only rings `DIAL_ALLOWLIST` numbers, and never a lead without consent, on the suppression list, archived, assigned to a rep, or with a salesperson's decision.
- A customer who says "don't call me" goes on the suppression list.
- The agent must say it is an AI in its first sentence (EU AI Act Art. 50) and only quote prices from `services.md`.
- Only allowlisted test phones until Vodafone authorises calls under its name.

## Commits

No Claude attribution lines in commit messages (see `CLAUDE.md`).
