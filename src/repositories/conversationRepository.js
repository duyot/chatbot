// Data-access layer for chat-history conversations. Uses apiFetch so the
// Authorization header and 401 handling are applied centrally.
import { apiFetch } from '../api/client'

export async function listConversations() {
  const res = await apiFetch('/api/conversations')
  if (!res.ok) throw new Error('Failed to load conversations')
  return res.json() // [{ id, title, created_at, updated_at }]
}

export async function getConversation(id) {
  const res = await apiFetch(`/api/conversations/${id}`)
  if (!res.ok) throw new Error('Failed to load conversation')
  return res.json() // { id, title, created_at, messages: [...] }
}

export async function deleteConversation(id) {
  const res = await apiFetch(`/api/conversations/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete conversation')
}

export async function clearConversations() {
  const res = await apiFetch('/api/conversations', { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to clear conversations')
}
