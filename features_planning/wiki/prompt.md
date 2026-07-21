Analyze this repository and generate concise project documentation under a `wiki/` folder.

## Step 1 — Explore before writing
- Read the entry points, config files (build files, docker-compose, env samples), and folder structure first.
- Trace the actual code paths — do not guess or invent flows that aren't in the code.
- If something is ambiguous, state it as "unclear/assumption" rather than presenting it as fact.

## Step 2 — Generate these files under wiki/

1. `wiki/README.md` — Wiki index with links to all pages, plus a 5-line project summary
   (what it does, who uses it, main tech stack).

2. `wiki/01-architecture.md`
   - High-level architecture diagram (Mermaid `graph TD`): components, external services,
     databases, queues, and how they connect.
   - One short paragraph per component: responsibility, key classes/modules, entry point file paths.

3. `wiki/02-flows.md` — For EVERY major flow in the system (API request handling, background
   jobs, scheduled tasks, event/message consumption, startup/init):
   - A Mermaid `sequenceDiagram` (mandatory — one per flow)
   - Below each diagram: a numbered step list with actual file:function references
     (e.g. `service/ingest.py:process_document()`)

4. `wiki/03-data-model.md` — Key tables/collections/schemas, their relationships
   (Mermaid `erDiagram` if applicable), and which code owns reads/writes.

5. `wiki/04-integrations.md` — External APIs, services, queues: what is called, where in code,
   auth method, and failure handling if visible.

6. `wiki/05-setup.md` — How to run locally: prerequisites, env vars (names only, no secrets),
   commands, and how to verify it's working.

## Constraints
- Be concise: prefer diagrams and tables over long prose. Each page ≤ ~150 lines.
- Every claim must be traceable to a file path — include the path.
- Diagrams must be Mermaid so they render in GitHub/IDE previews.
- Skip generated code, vendored dependencies, and test fixtures.
- After generating, list any parts of the codebase you did NOT cover.