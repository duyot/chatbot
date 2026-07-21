// Data-access layer for documents. Uses apiFetch so the Authorization header
// and 401 handling are applied centrally.
import { apiFetch } from '../api/client'

export async function listDocuments() {
  const res = await apiFetch('/api/documents')
  if (!res.ok) throw new Error('Failed to load documents')
  return res.json() // [{ id, file_name, uploaded_at, mime_type }]
}

export async function getDocumentFile(id) {
  const res = await apiFetch(`/api/documents/${id}/file`)
  if (!res.ok) throw new Error('Failed to load document file')
  return res.blob()
}
