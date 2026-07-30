import { useState, useEffect } from 'react'
import { getDocumentFile } from '../repositories/documentRepository'
import CitationOverlay from './CitationOverlay'
import PageImageViewer from './PageImageViewer'
import './FilePreview.css'

function CloseIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M2 2l9 9M11 2l-9 9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  )
}

export default function FilePreview({ document, onClose, highlightTarget = null }) {
  const mimeType = document?.mime_type || ''
  const isPdf = mimeType === 'application/pdf'

  // PDFs preview from server-rendered page images, so we never download the
  // source file — see PageImageViewer.
  const docId = document?.id || null
  const needsFile = Boolean(docId) && !isPdf

  // Single state object, tagged with the document it describes, so status and
  // url can never disagree and a stale blob can't be shown for a new document.
  const [file, setFile] = useState({ id: null, status: 'idle', url: null })

  // Reset during render, not in an effect — an effect would paint one frame of
  // the previous document's preview first.
  if (file.id !== docId) {
    setFile({ id: docId, status: needsFile ? 'loading' : 'idle', url: null })
  }

  useEffect(() => {
    if (!needsFile) return undefined

    let cancelled = false
    let url = null

    async function load() {
      try {
        const blob = await getDocumentFile(docId)
        if (cancelled) return
        url = URL.createObjectURL(blob)
        setFile({ id: docId, status: 'ready', url })
      } catch {
        if (!cancelled) setFile({ id: docId, status: 'error', url: null })
      }
    }

    load()

    return () => {
      cancelled = true
      if (url) URL.revokeObjectURL(url)
    }
  }, [docId, needsFile])

  if (!document) return null

  // Highlight target scoped to this document (images overlay directly; PDF
  // pages are overlaid per-page by PageImageViewer).
  const docHighlight =
    highlightTarget && highlightTarget.documentId === document.id ? highlightTarget : null
  const highlightRects = docHighlight?.rects || []
  const isImage = mimeType.startsWith('image/')

  return (
    <div className="file-preview">
      <div className="file-preview-header">
        <span className="file-preview-name">{document.file_name}</span>
        <button className="file-preview-close" title="Close preview" onClick={onClose}>
          <CloseIcon />
        </button>
      </div>
      <div className="file-preview-body">
        {isPdf && (
          // Keyed so switching documents mounts a fresh viewer, which is what
          // lets usePageImages avoid any reset logic of its own.
          <PageImageViewer
            key={document.id}
            documentId={document.id}
            highlightTarget={docHighlight}
          />
        )}
        {!isPdf && file.status === 'loading' && (
          <p className="file-preview-hint">Loading preview…</p>
        )}
        {!isPdf && file.status === 'error' && (
          <p className="file-preview-hint">Couldn&apos;t load a preview for this file.</p>
        )}
        {!isPdf && file.status === 'ready' && isImage && (
          <div className="file-preview-image-wrap">
            <img className="file-preview-image" src={file.url} alt={document.file_name} />
            <CitationOverlay rects={highlightRects} />
          </div>
        )}
        {!isPdf && file.status === 'ready' && !isImage && (
          <div className="file-preview-fallback">
            <p className="file-preview-hint">Preview not available for this file type.</p>
            <a className="file-preview-download" href={file.url} download={document.file_name}>
              Download {document.file_name}
            </a>
          </div>
        )}
      </div>
    </div>
  )
}
