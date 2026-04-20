# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

This repo is currently the **default Vite + React 19 scaffold** (from `create-vite`). `App.jsx` is still the starter page — no chatbot logic has been implemented yet. Treat feature work as greenfield on top of the scaffold.

## Stack

- **React 19.2** with `StrictMode` (see `src/main.jsx`)
- **Vite 8** as build tool and dev server (`vite.config.js` only registers `@vitejs/plugin-react`)
- **ESLint 9** flat config (`eslint.config.js`) — uses `@eslint/js` recommended, `eslint-plugin-react-hooks`, and `eslint-plugin-react-refresh` (Vite variant)
- **JavaScript (JSX)**, not TypeScript. No `tsconfig.json`, no type tooling.
- **No test framework installed yet.** Adding tests requires picking and installing one (e.g. Vitest + React Testing Library, or Playwright for E2E).

## Commands

```bash
npm run dev       # Start Vite dev server with HMR
npm run build     # Production build to dist/
npm run preview   # Preview the production build
npm run lint      # ESLint over the whole repo
```

To lint a single file: `npx eslint path/to/file.jsx`.

## ESLint notes

The config in `eslint.config.js` has one non-default rule worth knowing about:

```js
'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }]
```

Unused variables are errors **unless** they start with an uppercase letter or underscore. This exists so unused imports of React component types / constants don't break the lint, but unused lowercase locals will fail CI/lint.

## Architecture

Entry path is standard Vite:

- `index.html` → loads `/src/main.jsx`
- `src/main.jsx` mounts `<App />` inside `StrictMode` on `#root`
- `src/App.jsx` is the current (scaffold) root component
- `public/` holds static assets served from `/` (e.g. `/icons.svg`, `/favicon.svg`) — reference them by absolute path in JSX, not via `import`

When the chatbot UI is built, follow the common layout that matches the existing rules:
- Keep components in `src/` split into small files (≤400 lines); extract hooks into their own files.
- Components that fetch or mutate data should go through a repository-style abstraction rather than calling `fetch` inline — see the repository pattern in the user's global rules.

## Working in this repo

- This is a **`.jsx` project, not `.tsx`**. The global TypeScript rules in the user's config describe the preferred style, but apply them via JSDoc annotations where helpful rather than adding TypeScript unilaterally. Ask before introducing TypeScript, since it would mean adding `tsconfig.json`, a type-aware ESLint setup, and converting files.
- There is no `.env` handling wired up. Any API keys (OpenAI, Anthropic, etc.) for the chatbot must go through `import.meta.env.VITE_*` with Vite's env loading — do not hardcode.
- Vite config is intentionally minimal. Before adding plugins (proxy, path aliases, PWA, etc.), discuss the trade-off first.
- There is no Git repo initialized in this working directory and no CI config. Don't assume either exists when writing commands or hooks.