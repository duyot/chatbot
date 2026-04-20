import { useState, useRef } from 'react'
import './Composer.css'

export default function Composer({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

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

  const canSend = value.trim().length > 0 && !disabled

  return (
    <div className="composer-wrapper">
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
