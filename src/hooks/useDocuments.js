import { useState, useEffect, useCallback } from 'react'
import { listDocuments } from '../repositories/documentRepository'

export function useDocuments() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments())
    } catch {
      /* ignore fetch errors; keep the existing list */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    // Initial load + refresh on window focus; refresh sets state internally.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refresh()
    window.addEventListener('focus', refresh)
    return () => window.removeEventListener('focus', refresh)
  }, [refresh])

  return { documents, loading, refresh }
}
