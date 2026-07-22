# Plan: Username/Password Login Gating the Chat

**Complexity:** MEDIUM (full-stack: DB + auth backend + React auth flow)

## Requirements
1. From the home page, "Start chat" must route through a login screen — chat is no longer directly accessible.
2. Users authenticate with username + password before reaching `/chat`.
3. Provide a script to create the `users` table.
4. Implement authentication against the `users` table.

## Decisions (confirmed)
| Decision | Choice |
|---|---|
| Session mechanism | JWT bearer token in `Authorization` header, stored in `localStorage` |
| User provisioning | Seed CLI `scripts/create_user.py` (no public signup) |
| Enforcement | Protect backend `/api/*` too — EXCEPT `/api/documents/{id}/status` (EventSource) stays public |

## Backend
- `models.py`: `User` (`id`, `username` unique, `password_hash`, `is_active`, `created_at`).
- `alembic/versions/0007_create_users.py`: create `users` + unique index on `username`.
- `scripts/create_users_table.sql`: raw DDL (literal "script to create table users").
- `scripts/create_user.py`: seed CLI — `python -m scripts.create_user --username X --password Y`.
- `security.py`: bcrypt hash/verify, JWT mint/decode, `get_current_user` dependency.
- `routers/auth.py`: `POST /api/auth/login`, `GET /api/auth/me`.
- `config.py`: `jwt_secret_key`, `jwt_algorithm`, `access_token_expire_minutes`.
- `main.py`: register auth router.
- `routers/chat.py` + `routers/documents.py` (list + upload): require `get_current_user`.
- `requirements.txt`: `PyJWT`, `passlib[bcrypt]`, pinned `bcrypt`.
- `tests/test_auth.py`: login success/fail, protected route 401/200.

## Frontend
- `src/api/client.js`: `apiFetch` injecting `Authorization: Bearer`, clears token + emits `auth:unauthorized` on 401.
- `src/repositories/authRepository.js`: `login()`, `getMe()`.
- `src/auth/AuthContext.jsx`: `AuthProvider` + `useAuth` (token/user, persistence, refresh-time validation).
- `src/components/ProtectedRoute.jsx`: redirect to `/login` (preserving intended path).
- `src/pages/LoginPage.jsx` + `.css`: username/password form.
- `src/App.jsx`: wrap in `AuthProvider`; add `/login`; guard `/chat`.
- `useChat.js`, `useUpload.js` (upload only), `DocumentSelector.jsx`: send token via `apiFetch`.
- HomePage unchanged — "Start chat" → `/chat`; the guard redirects logged-out users to `/login`.

## Validation
```bash
cd backend && pip install -r requirements.txt
alembic upgrade head
python -m scripts.create_user --username alice --password secret123
pytest tests/test_auth.py
cd .. && npm run lint && npm run build
```
