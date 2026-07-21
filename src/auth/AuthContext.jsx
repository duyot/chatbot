import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { login as loginRequest, getMe } from '../repositories/authRepository'
import { getToken, setToken } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(() => getToken())
  const [user, setUser] = useState(null)
  // While we validate a token found in storage on first load, hold routing.
  const [loading, setLoading] = useState(() => Boolean(getToken()))

  // Validate an existing token (e.g. after a page refresh) and hydrate user.
  useEffect(() => {
    let cancelled = false
    if (!token) {
      // No token → nothing to validate; `loading` already initialised false.
      return
    }
    getMe(token)
      .then((u) => {
        if (!cancelled) setUser(u)
      })
      .catch(() => {
        if (!cancelled) {
          setToken(null)
          setTokenState(null)
          setUser(null)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [token])

  // apiFetch fires this when the API rejects our token mid-session.
  useEffect(() => {
    const handler = () => {
      setTokenState(null)
      setUser(null)
    }
    window.addEventListener('auth:unauthorized', handler)
    return () => window.removeEventListener('auth:unauthorized', handler)
  }, [])

  const login = useCallback(async (username, password) => {
    const { access_token } = await loginRequest({ username, password })
    setToken(access_token)
    setTokenState(access_token)
    const u = await getMe(access_token).catch(() => ({ username }))
    setUser(u)
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    setTokenState(null)
    setUser(null)
  }, [])

  const value = {
    token,
    user,
    loading,
    isAuthenticated: Boolean(token),
    login,
    logout,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider')
  return ctx
}
