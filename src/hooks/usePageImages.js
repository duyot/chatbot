import { useCallback, useEffect, useRef, useState } from 'react'
import { listDocumentPages, getDocumentPageImage } from '../repositories/documentRepository'

/**
 * Loads a document's server-rendered preview pages.
 *
 * The manifest arrives up front (cheap — just dimensions), but page bytes are
 * fetched on demand via `requestPage` so opening a 200-page document doesn't
 * pull 200 images. Images go through `apiFetch` rather than a bare `<img src>`
 * because the route is authenticated and an `<img>` cannot send headers; each
 * blob becomes an object URL, revoked on unmount.
 *
 * `documentId` is expected to be stable for the lifetime of the consuming
 * component — render it with `key={documentId}` so switching documents mounts a
 * fresh instance. That keeps every reset path out of this hook.
 *
 * @param {string|undefined} documentId
 * @returns {{
 *   pages: { page: number, width: number, height: number }[],
 *   urls: Record<number, string>,
 *   loading: boolean,
 *   error: boolean,
 *   requestPage: (page: number) => void,
 * }}
 */
export function usePageImages(documentId) {
  const [manifest, setManifest] = useState(() => ({
    status: documentId ? 'loading' : 'idle',
    pages: [],
  }))
  const [urls, setUrls] = useState({})

  // Object URLs we own, so unmount cleanup can revoke them without depending on
  // render state. Touched only from callbacks and effects.
  const urlsRef = useRef({})
  // Pages already in flight, so overlapping observer callbacks don't request the
  // same page twice.
  const pendingRef = useRef(new Set())

  useEffect(() => {
    if (!documentId) return undefined

    let cancelled = false
    listDocumentPages(documentId)
      .then((pages) => {
        if (!cancelled) setManifest({ status: 'ready', pages })
      })
      .catch(() => {
        if (!cancelled) setManifest({ status: 'error', pages: [] })
      })

    return () => { cancelled = true }
  }, [documentId])

  // Release every object URL we still hold when the viewer goes away.
  useEffect(() => () => {
    Object.values(urlsRef.current).forEach((url) => URL.revokeObjectURL(url))
    urlsRef.current = {}
  }, [])

  const requestPage = useCallback((page) => {
    if (!documentId || !page) return
    if (urlsRef.current[page] || pendingRef.current.has(page)) return

    pendingRef.current.add(page)

    getDocumentPageImage(documentId, page)
      .then((blob) => {
        const url = URL.createObjectURL(blob)
        urlsRef.current = { ...urlsRef.current, [page]: url }
        setUrls((prev) => ({ ...prev, [page]: url }))
      })
      .catch(() => {
        // One unreachable page must not break the rest of the preview; that
        // page's placeholder simply stays empty.
      })
      .finally(() => {
        pendingRef.current.delete(page)
      })
  }, [documentId])

  return {
    pages: manifest.pages,
    urls,
    loading: manifest.status === 'loading',
    error: manifest.status === 'error',
    requestPage,
  }
}
