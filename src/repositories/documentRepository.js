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

/**
 * Manifest of server-rendered preview pages, in page order. Empty for file
 * types that have no page images (single images, DOCX).
 *
 * @param {string} id
 * @returns {Promise<{ page: number, width: number, height: number }[]>}
 */
export async function listDocumentPages(id) {
  const res = await apiFetch(`/api/documents/${id}/pages`)
  if (!res.ok) throw new Error('Failed to load document pages')
  return res.json()
}

/**
 * One rendered page image. Goes through apiFetch (not a bare <img src>) because
 * the route is authenticated and an <img> cannot send the Authorization header.
 *
 * @param {string} id
 * @param {number} page 1-based
 * @returns {Promise<Blob>}
 */
export async function getDocumentPageImage(id, page) {
  const res = await apiFetch(`/api/documents/${id}/pages/${page}`)
  if (!res.ok) throw new Error(`Failed to load page ${page}`)
  return res.blob()
}
