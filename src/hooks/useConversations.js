import { useState, useEffect, useCallback } from 'react'
import {
  listConversations,
  deleteConversation,
  clearConversations,
} from '../repositories/conversationRepository'

export function useConversations() {
  const [conversations, setConversations] = useState([])
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      setConversations(await listConversations())
    } catch {
      /* ignore fetch errors; keep the existing list */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    // Initial load. refresh sets state internally.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refresh()
  }, [refresh])

  const remove = useCallback(async (id) => {
    await deleteConversation(id)
    setConversations((cs) => cs.filter((c) => c.id !== id))
  }, [])

  const clearAll = useCallback(async () => {
    await clearConversations()
    setConversations([])
  }, [])

  return { conversations, loading, refresh, remove, clearAll }
}
