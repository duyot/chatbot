// Data-access layer for authentication endpoints. Components/hooks call these
// instead of using fetch inline.

export async function login({ username, password }) {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    let detail = 'Login failed'
    try {
      detail = (await res.json()).detail || detail
    } catch {
      /* non-JSON error body; keep default */
    }
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  return res.json() // { access_token, token_type }
}

export async function getMe(token) {
  const res = await fetch('/api/auth/me', {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) throw new Error('Not authenticated')
  return res.json() // { id, username }
}
