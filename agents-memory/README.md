# agents-memory/

Token-cheap working context for a **coding agent** on this repo. Read
`memory.md` first each session. Not product docs.

**agents-memory vs spec:** `spec/` = *what* the repo does + *why* + use cases
(human/product). `agents-memory/` = *how to edit/run it safely* (agent). When
the *what/why* is needed, **link the spec section — never copy it**.

- `memory.md` — durable agent context: repo map, edit invariants, run/test,
  pitfalls. Terse; each line points at a spec doc. Changes slowly.
- `sessions/` — one file per session, `YYYY-MM-DD-<topic>.md`, append-only
  (Goal / Context / Done+why / Follow-ups / Status). Promote durable facts up
  into `memory.md`; don't rewrite history.

Keep every file here short — it is paid for on every session load.
