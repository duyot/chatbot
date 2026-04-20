import { useEffect } from 'react'
import './UploadToast.css'

export default function UploadToast({ status, fileName, error, onDismiss }) {
  useEffect(() => {
    if (status !== 'done' && status !== 'failed') return
    const t = setTimeout(onDismiss, 5000)
    return () => clearTimeout(t)
  }, [status, onDismiss])

  if (status !== 'done' && status !== 'failed') return null

  const isDone = status === 'done'

  return (
    <div className={`upload-toast upload-toast--${isDone ? 'done' : 'failed'}`} role="alert">
      <div className="upload-toast-icon" aria-hidden="true">
        {isDone ? (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M2 5l2.5 2.5L8 3" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M2 2l6 6M8 2l-6 6" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        )}
      </div>
      <div className="upload-toast-body">
        <div className="upload-toast-title">{isDone ? 'Document ready' : 'Ingestion failed'}</div>
        <div className="upload-toast-msg">
          {isDone
            ? `${fileName} has been ingested and is ready for Q&A.`
            : (error || 'Something went wrong.')}
        </div>
      </div>
      <button className="upload-toast-close" onClick={onDismiss} aria-label="Dismiss">✕</button>
    </div>
  )
}
