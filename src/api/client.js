// Central place for talking to the backend API.
//
// - Stores the JWT in localStorage under TOKEN_KEY.
// - apiFetch() injects the Authorization header and, on a 401, clears the
//   token and dispatches an `auth:unauthorized` window event so AuthContext
//   can log the user out and bounce them to /login.

export const TOKEN_KEY = 'chatbot_token'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
}

export function authHeaders(extra = {}) {
  const token = getToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : { ...extra }
}

export async function apiFetch(input, init = {}) {
  const res = await fetch(input, { ...init, headers: authHeaders(init.headers) })
  if (res.status === 401) {
    setToken(null)
    window.dispatchEvent(new Event('auth:unauthorized'))
  }
  return res
}
