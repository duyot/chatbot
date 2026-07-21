# Plan: Chat History + AI Response Animation

**Complexity**: Medium (History) + Small (Animation)
**Mode**: Conversational plan, grounded in the current codebase

This file covers the two tasks captured below. **Feature 1 (Chat History)** is the primary, detailed plan. **Feature 2 (AI Response Animation)** is a smaller, mostly-frontend enhancement planned in its own section.

---

## Original Requirements

### Task 1 — chat history functionality
Current:
1. The history in the left side of the chat is hardcoded HTML, not real conversations.
2. A `users` table was just added to store the login user.

Requirement:
- a. Store a new conversation on each new chat, mapped to the current logged-in user.
- b. Store every user prompt and AI response within a conversation.
- c. Load all of the user's conversations in the left sidebar.
- d. On clicking a conversation, load all its user + AI messages back into the chat.

### Task 2 — enhance the AI response animation
Current:
1. After the user submits a prompt there is no in-progress response/animation in the UI.
2. This makes the UX poor.

Task:
1. After submit, show a loading / "thinking…" animation so it feels nicer.
2. Follow the Claude Code style — messages can cycle: Processing, Ingesting, Sprouting… with animation.
3. If it takes a long time, show "this may take longer than expected…".

## Confirmed Decisions (Task 1)

1. **Switchable document per message** — a conversation is *not* locked to one document. Each message stores its own `document_id`. On reload, the document selector is best-effort set to the *last* message's document so the user can keep chatting, but they can switch freely.
2. **Extra sidebar actions**: **Delete conversation** and **Clear All** are in scope. Rename is out of scope this pass.

---

# Feature 1 — Chat History

## Summary

The chat stream endpoint persists nothing today and the sidebar renders hardcoded fake conversations + a fake user. We add two tables (`conversations`, `messages`), make `POST /api/chat/stream` create/attach a conversation and persist the user prompt + accumulated AI answer (with citations), add a user-scoped `conversations` router (list / detail / delete / clear-all), and rewire the frontend sidebar + chat hook to load real conversations and restore messages on click.

## Patterns to Mirror

| Category | Source | Pattern |
|---|---|---|
| Model | `backend/app/models.py:72` (`User`) | UUID PK `default=uuid.uuid4`, `DateTime(timezone=True)` w/ `lambda: datetime.now(timezone.utc)`, `ForeignKey(..., ondelete=...)`, `Index` in `__table_args__` |
| Migration | `backend/alembic/versions/0007_create_users.py` | Numbered `000N_*.py`, `revision`/`down_revision`, `server_default=sa.text("gen_random_uuid()")`, explicit `create_index`/`drop_index` |
| Router | `backend/app/routers/documents.py:20` | `APIRouter(prefix="/api/...")`, `Depends(get_db)`+`Depends(get_current_user)`, `response_model`, `raise HTTPException`, `logger.info` |
| Schema | `backend/app/schemas.py:37` (`UserResponse`) | Pydantic v2, `model_config = {"from_attributes": True}` |
| Tests | `backend/tests/test_auth.py` + `conftest.py` | `client`/`db` fixtures, transactional rollback, `_make_user`, login→token→request, assert 401 without token |
| FE data access | `src/repositories/authRepository.js` + `src/api/client.js` | Repo module of async fns; use `apiFetch` (injects auth, handles 401) for authed calls |
| FE hook | `src/hooks/useChat.js` | `useReducer` in own file, SSE parse loop, action per event |
| FE auth state | `src/auth/AuthContext.jsx:76` | `useAuth()` → `{ user: { id, username } }` |

## Data Model

```
conversations
  id          UUID PK
  user_id     UUID FK users.id (ON DELETE CASCADE)          [indexed]
  title       String   (first user message, truncated ~60 chars)
  created_at  timestamptz
  updated_at  timestamptz  (bumped on each new message -> sidebar ordering)

messages
  id              UUID PK
  conversation_id UUID FK conversations.id (ON DELETE CASCADE) [indexed]
  document_id     UUID FK documents.id (ON DELETE SET NULL, nullable)
  role            String   ('user' | 'assistant')
  content         Text
  citations       JSONB    (nullable; the `chunks` array for assistant messages)
  created_at      timestamptz
```

## Files to Change

| File | Action | Why |
|---|---|---|
| `backend/app/models.py` | UPDATE | Add `Conversation` + `Message` models |
| `backend/alembic/versions/0008_create_conversations_messages.py` | CREATE | Migration for both tables (down_revision `0007`) |
| `backend/app/schemas.py` | UPDATE | `ConversationListItem`, `MessageResponse`, `ConversationDetail`; add `conversation_id` to `ChatRequest` |
| `backend/app/routers/conversations.py` | CREATE | `GET /api/conversations`, `GET /{id}`, `DELETE /{id}`, `DELETE /` — all user-scoped |
| `backend/app/routers/chat.py` | UPDATE | Create/attach conversation, persist user + assistant msgs, emit `conversation` event |
| `backend/app/main.py` | UPDATE | `include_router(conversations.router)` |
| `backend/tests/test_conversations.py` | CREATE | Ownership isolation, auth-required, ordering, delete, clear-all, stream persistence (mocked RAG) |
| `src/repositories/conversationRepository.js` | CREATE | `listConversations`, `getConversation`, `deleteConversation`, `clearConversations` |
| `src/hooks/useConversations.js` | CREATE | Load/refresh list, `remove(id)`, `clearAll()` |
| `src/hooks/useChat.js` | UPDATE | Track `conversationId`, handle `conversation` event, add `loadMessages`/`reset`, send `conversation_id` |
| `src/components/Sidebar.jsx` | UPDATE | Real conversations + `useAuth` user; wire New chat / select / delete / Clear All |
| `src/pages/ChatPage.jsx` | UPDATE | Orchestrate `useConversations` + `useChat`; select→load, new-chat→reset, created→refresh |

## Backend API Contract

- `POST /api/chat/stream` (modified) — body `{ document_id, message, conversation_id? }`.
  - No `conversation_id` → create `Conversation` for `current_user`, title = first message truncated. Emit `{"type":"conversation","conversation_id":"...","title":"..."}` **before** tokens.
  - With `conversation_id` → verify it belongs to `current_user` (else `error` event).
  - Persist user message; stream via `agentic_rag_stream`; accumulate `token` content + `citations.chunks`; persist assistant message; bump `conversation.updated_at`. All writes use the generator's `SessionLocal()` (`stream_db`); capture `current_user.id` into a local before the generator closure.
- `GET /api/conversations` → `[{ id, title, created_at, updated_at }]`, current user only, `updated_at` desc.
- `GET /api/conversations/{id}` → `{ id, title, created_at, messages: [{ id, role, content, citations, document_id, created_at }] }`, ownership-checked (404 otherwise), messages `created_at` asc.
- `DELETE /api/conversations/{id}` → 204, ownership-checked.
- `DELETE /api/conversations` → 204, deletes all of current user's conversations (Clear All).

## Tasks

### T1 — Models
- Add `Conversation` + `Message` to `models.py`. **Mirror**: `User`. **Validate**: `cd backend && python -c "from app.models import Conversation, Message"`.

### T2 — Migration
- `0008_create_conversations_messages.py`: both tables + indexes on `conversations.user_id`, `messages.conversation_id`; full `downgrade()`. **Mirror**: `0007`. **Validate**: `alembic upgrade head` then `alembic downgrade -1 && alembic upgrade head`.

### T3 — Schemas
- Add the 3 response schemas; add `conversation_id: str | None = None` to `ChatRequest`. **Mirror**: `UserResponse`.

### T4 — Conversations router
- Implement list/detail/delete/clear-all scoped to `current_user`; register in `main.py`. **Mirror**: `documents.py`. **Validate**: T7 tests.

### T5 — Persist in chat stream
- Modify `chat.py` per contract above. **Mirror**: existing `event_stream()` + `SessionLocal()`. **Validate**: T7 stream test.

### T6 — Frontend wiring
- `conversationRepository.js` + `useConversations.js`; extend `useChat.js` (track/emit `conversationId`, `loadMessages`, `reset`, send `conversation_id`); rewrite `Sidebar.jsx` (real data + real user, wire actions); orchestrate in `ChatPage.jsx` (select→`getConversation`→`loadMessages` + set selector to last message's doc; new-chat→`reset`; created→refresh + mark active). **Mirror**: `authRepository.js`, `apiFetch`, `useChat` reducer. **Validate**: `npm run lint && npm run build` + manual browser check.

### T7 — Backend tests
- `test_conversations.py`: auth-required (401); list ownership isolation; detail ordering + 404 for other user; delete + 404 for other user; clear-all scoped; stream persistence with `monkeypatch` on `agentic_rag_stream` yielding fake `token`/`citations`/`done` → assert 1 conversation + user + assistant messages persisted and `conversation` event emitted. **Mirror**: `test_auth.py`. **Validate**: `pytest tests/test_conversations.py -v`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stream errors mid-answer → assistant not persisted | Medium | Persist user msg up front; save assistant only on clean completion (partial answers not saved). |
| Two DB sessions in stream endpoint | Medium | All writes in `stream_db`; capture `current_user.id` before the generator. |
| Missing ownership check leaks another user's history | High if missed | Filter every endpoint by `current_user.id`; 404 when not owned; explicit isolation tests. |
| FE message-key collision (DB uuids vs int ids) | Low | Loaded msgs keyed by DB `id`; `makeId()` only for in-flight new msgs. |
| Stale sidebar order after reply | Low | Bump `updated_at` on assistant save; list ordered `updated_at` desc; refresh on `conversation` event. |
| Document deleted → dangling `document_id` | Low | FK `ON DELETE SET NULL`; FE tolerates null doc on restore. |

---

# Feature 2 — AI Response Animation

## Summary

Today, `useChat` dispatches `START_ASSISTANT_MESSAGE` immediately on send, so `ChatThread`'s `pending && !anyStreaming` typing indicator (`ChatThread.jsx:26`) **never renders** — during the multi-second retrieval phase (rewrite → hybrid retrieval → rerank → grade → generate) the assistant bubble shows only a bare blinking cursor (`ChatMessage.jsx:80`). The RAG stream emits no per-node phase events (only `token` / `warning` / `citations` / `done` — `graph.py:91`), so a **frontend-only** cycling "thinking" animation is the right fit and matches the whimsical-label requirement.

## Approach (frontend-only, no backend change)

Render an animated status indicator inside the streaming assistant bubble while it is `streaming && content === ''`. Once the first token arrives, it disappears and normal streaming text + cursor takes over.

- New component `src/components/ThinkingIndicator.jsx` (+ `.css`): cycles labels (`Processing…`, `Ingesting…`, `Sprouting…`) on a ~2.5s interval with a subtle animation (fade + the existing dot pulse style from `ChatThread.css`).
- Tracks elapsed time; after a threshold (~15s) switches/appends "This may take longer than expected…".
- Self-contained timers (`useEffect` + interval) so it cleans up on unmount when tokens start.

## Files to Change

| File | Action | Why |
|---|---|---|
| `src/components/ThinkingIndicator.jsx` | CREATE | Cycling label + long-wait message + animation |
| `src/components/ThinkingIndicator.css` | CREATE | Animation styles (reuse dot-pulse look) |
| `src/components/ChatMessage.jsx` | UPDATE | When assistant + `streaming` + empty `content`, render `<ThinkingIndicator />` instead of bare cursor |
| `src/components/ChatThread.jsx` | UPDATE (optional) | Remove now-dead `pending && !anyStreaming` block, or leave as harmless fallback |

## Tasks

### A1 — ThinkingIndicator component
- Build the component with a label array, interval cycling, and elapsed-time threshold for the long-wait message. **Mirror**: dot-pulse markup/animation in `ChatThread.jsx:26` + `ChatThread.css`.

### A2 — Wire into the streaming bubble
- In `ChatMessage.jsx`, when `!isUser && streaming && !content.trim()`, render `<ThinkingIndicator />`; otherwise render content + cursor as today. **Validate**: `npm run lint && npm run build` + manual: send a prompt and watch the labels cycle until the answer streams.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Interval not cleared on unmount → warning/leak | Low | `clearInterval` in `useEffect` cleanup. |
| Labels flash if the answer starts instantly | Low | Only render while `content` empty; short first-label delay acceptable. |
| `prefers-reduced-motion` users | Low | Gate the pulse animation behind `@media (prefers-reduced-motion: no-preference)`. |

## Optional future accuracy (not this pass)
Emit real backend phase events from `agentic_rag_stream` (hook `on_chain_start`/`on_chain_end` per node in `astream_events`, map node → friendly label) and drive the indicator from real phases instead of a timer.

---

## Combined Validation

```bash
# Backend
cd backend
alembic upgrade head
pytest tests/test_conversations.py -v
pytest -m "not eval"            # full suite still green

# Frontend
cd ..
npm run lint
npm run build
# Manual: login -> new chat -> ask across 2 documents (watch the thinking animation) ->
#   reload page -> conversation in sidebar -> click it -> messages + citations restore ->
#   delete one -> Clear All
```

## Acceptance

- [ ] New chat persists a `Conversation` mapped to the logged-in user on first send.
- [ ] Every user prompt and AI response (with citations) is stored under that conversation.
- [ ] Sidebar lists the current user's real conversations, most-recent first, real username in footer.
- [ ] Clicking a conversation restores all messages + citations and continues in the same conversation.
- [ ] Delete and Clear All work, scoped to the current user.
- [ ] After submit, an animated "thinking" indicator cycles playful labels until the answer streams; a long-wait message appears after the threshold.
- [ ] `pytest tests/test_conversations.py` green; `npm run lint` and `npm run build` pass.
- [ ] Patterns mirrored (models/migration/router/schema/repository), not reinvented.

## Suggested Build Order
T1 → T2 → T3 → T4 → T7 (backend green) → T5 → T6 (history end-to-end) → A1 → A2 (animation).
