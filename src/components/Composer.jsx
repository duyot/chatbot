import { useState, useRef } from 'react'
import './Composer.css'

export default function Composer({ onSend, onFileSelect, uploadStatus, disabled }) {
  const [value, setValue] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.focus()
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e)
    }
  }

  function handleChange(e) {
    setValue(e.target.value)
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 120) + 'px'
    }
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    onFileSelect(file)
    e.target.value = ''
  }

  function handleRemoveFile() {
    setSelectedFile(null)
  }

  const isUploading = uploadStatus === 'uploading' || uploadStatus === 'processing'
  const canSend = value.trim().length > 0 && !disabled

  return (
    <div className="composer-wrapper">
      {selectedFile && (
        <div className="composer-chip-row">
          <div className="composer-chip">
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <rect x="2" y="1" width="10" height="14" rx="2" stroke="#0071e3" strokeWidth="1.5" />
              <path d="M5 5h6M5 8h4" stroke="#0071e3" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <span className="composer-chip-name">{selectedFile.name}</span>
            {isUploading && <span className="composer-chip-spinner" aria-label="Uploading" />}
            {!isUploading && (
              <button className="composer-chip-remove" onClick={handleRemoveFile} aria-label="Remove file">✕</button>
            )}
          </div>
        </div>
      )}
      <form className="composer" onSubmit={handleSubmit}>
        <div className="composer-model-icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 18 18">
            <defs>
              <linearGradient id="mg" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#0071e3" />
                <stop offset="100%" stopColor="#2997ff" />
              </linearGradient>
            </defs>
            <circle cx="9" cy="9" r="9" fill="url(#mg)" />
            <circle cx="9" cy="9" r="4" fill="white" opacity="0.85" />
          </svg>
        </div>
        <textarea
          ref={textareaRef}
          className="composer-input"
          placeholder="What's in your mind?..."
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={disabled}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,image/png,image/jpeg,image/webp"
          className="composer-file-input"
          onChange={handleFileChange}
          aria-label="Upload file"
        />
        <button
          type="button"
          className={`composer-upload ${isUploading ? 'composer-upload--busy' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          aria-label="Upload file"
          title="Upload PDF, DOCX, or image"
        >
          {isUploading ? (
            <span className="composer-upload-spinner" />
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 11v2a1 1 0 001 1h10a1 1 0 001-1v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M8 2v8M5 5l3-3 3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>
        <button
          type="submit"
          className={`composer-send ${canSend ? 'composer-send--active' : ''}`}
          disabled={!canSend}
          aria-label="Send"
        >
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M9 14V4M9 4L5 8M9 4L13 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </form>
    </div>
  )
}
