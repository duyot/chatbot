import { useState, useEffect } from 'react'
import { getDocumentFile } from '../repositories/documentRepository'
import CitationOverlay from './CitationOverlay'
import PdfViewer from './PdfViewer'
import './FilePreview.css'

function CloseIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M2 2l9 9M11 2l-9 9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  )
}

export default function FilePreview({ document, onClose, highlightTarget = null }) {
  const [blobUrl, setBlobUrl] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    if (!document?.id) return undefined

    let cancelled = false
    let url = null

    async function load() {
      setLoading(true)
      setError(false)
      setBlobUrl(null)
      try {
        const blob = await getDocumentFile(document.id)
        if (cancelled) return
        url = URL.createObjectURL(blob)
        setBlobUrl(url)
      } catch {
        if (!cancelled) setError(true)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()

    return () => {
      cancelled = true
      if (url) URL.revokeObjectURL(url)
    }
  }, [document?.id])

  if (!document) return null

  const mimeType = document.mime_type || ''
  // Highlight target scoped to this document (images overlay directly; PDFs are
  // rendered + overlaid by PdfViewer).
  const docHighlight =
    highlightTarget && highlightTarget.documentId === document.id ? highlightTarget : null
  const highlightRects = docHighlight?.rects || []

  return (
    <div className="file-preview">
      <div className="file-preview-header">
        <span className="file-preview-name">{document.file_name}</span>
        <button className="file-preview-close" title="Close preview" onClick={onClose}>
          <CloseIcon />
        </button>
      </div>
      <div className="file-preview-body">
        {loading && <p className="file-preview-hint">Loading preview…</p>}
        {!loading && error && (
          <p className="file-preview-hint">Couldn't load a preview for this file.</p>
        )}
        {!loading && !error && blobUrl && mimeType.startsWith('image/') && (
          <div className="file-preview-image-wrap">
            <img className="file-preview-image" src={blobUrl} alt={document.file_name} />
            <CitationOverlay rects={highlightRects} />
          </div>
        )}
        {!loading && !error && blobUrl && mimeType === 'application/pdf' && (
          <PdfViewer blobUrl={blobUrl} highlightTarget={docHighlight} />
        )}
        {!loading && !error && blobUrl && !mimeType.startsWith('image/') && mimeType !== 'application/pdf' && (
          <div className="file-preview-fallback">
            <p className="file-preview-hint">Preview not available for this file type.</p>
            <a className="file-preview-download" href={blobUrl} download={document.file_name}>
              Download {document.file_name}
            </a>
          </div>
        )}
      </div>
    </div>
  )
}
