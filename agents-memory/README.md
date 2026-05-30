# agents-memory/

AI-agent session notes and durable project memory for this repo, stored as
plain Markdown so any agent (or human) can read them before starting work.

## What lives here

- **`memory.md`** — durable, cross-session facts about the project: what it is,
  where it runs, the key entry points, the datastores, and the non-obvious
  gotchas. Read this first, every session. It should change slowly and only when
  a fact about the project becomes durably true (or stops being true).

- **`sessions/`** — one Markdown file per working session, recording **what was
  done and why**. These are an append-only log; you add a new file per session
  rather than rewriting old ones.

## Conventions

### Session notes (`sessions/`)
- **Naming:** `YYYY-MM-DD-<topic>.md` (e.g. `2026-05-29-refactor.md`). Date is
  the day the session started; `<topic>` is a short kebab-case slug.
- **Contents:** a factual record of a single session — typically Goal, Context,
  Plan, and Status. Capture *why* a decision was made, not just *what* changed;
  the diff already shows the what. Note loose ends and follow-ups so the next
  session can pick up.
- Session notes are immutable history. Don't rewrite a past session to reflect
  later facts — promote durable facts into `memory.md` instead.

### Durable memory (`memory.md`)
- Holds facts that outlive any single session and that an agent should know
  *before* touching the code: architecture, target hardware, entry points,
  datastores, and gotchas.
- Keep it concise and current. When a session establishes a new durable fact,
  fold it into `memory.md`; when a fact becomes false, fix it.

### Relationship to human docs
`learning.md` (repo root) and `spec/` are the human-facing documentation and
hold the detailed reasoning. `agents-memory/` is the agent's quick-start
context and working log — it points at those docs rather than duplicating them.
