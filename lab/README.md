[README](../README.md) | [Contributing](../CONTRIBUTING.md) | [Plans](./plans/README.md) | [Journal](./journal/README.md) | [Decisions](./decisions/README.md) | [Templates](./templates/README.md)

# Lab

This directory is the project's working notebook.

Use it to capture meaningful directions, experiments, decisions, and course corrections without turning the repo into a giant archive. Notes should stay tight, concrete, and linked.

## What Goes Where

- [Plans](./plans/README.md): forward-looking, scoped work plans with exit criteria.
- [Journal](./journal/README.md): dated experiment notes, findings, dead ends, and corrections.
- [Decisions](./decisions/README.md): stable conclusions contributors should not have to rediscover.
- [Templates](./templates/README.md): starter files for new lab notes.

## Rules

- Every markdown file should have a top navigation block.
- Every leaf document should link back up and link sideways where relevant.
- Every new lab document should be linked from its parent index.
- If a note is superseded, say so near the top and link to the replacement.
- Prefer concise sections and direct links to code, tests, plans, or related notes over long prose.

## Status Labels

- `active`: ongoing and expected to change.
- `validated`: a finding held up and can guide follow-on work.
- `accepted`: a decision is in force unless later superseded.
- `discarded`: the direction was explored and intentionally not pursued.
- `superseded`: replaced by a newer note.

## See Also

- [Current Implementation Plan](./plans/2026-05-15-implementation-plan.md)
- [Initial Repo Read And Planning Journal](./journal/2026-05-15-initial-repo-read-and-planning.md)
- [OpenAI Seed Shape For M2](./journal/2026-05-15-openai-seed-shape.md)
- [Claude Seed And Typed Model Contract](./journal/2026-05-15-claude-seed-and-typed-models.md)
- [Gemini Seed And Extraction Checkpoint](./journal/2026-05-15-gemini-seed-and-extraction-checkpoint.md)
- [Stabilization: Secrets, Backon, And Constructor Shape](./journal/2026-05-15-stabilization-secrets-and-backon.md)
- [M3 Multi-Provider Embedding Direction](./journal/2026-05-15-m3-multi-provider-embedding.md)
- [M4 Judge Layer Corrections](./journal/2026-05-15-m4-judge-layer.md)
- [M5a Two-Phase Consortium Orchestrator](./journal/2026-05-15-m5a-two-phase-orchestrator.md)
- [Decision 0001: Owned Types At Async Boundaries](./decisions/0001-owned-types-at-async-boundaries.md)
