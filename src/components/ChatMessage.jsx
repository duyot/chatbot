import { useState } from 'react'
import './ChatMessage.css'

function renderContent(content) {
  // Bold **text**, numbered lists
  const lines = content.split('\n')
  const elements = []
  let listItems = []
  let key = 0

  function flushList() {
    if (listItems.length > 0) {
      elements.push(
        <ol key={key++} className="msg-list">
          {listItems.map((li, i) => (
            <li key={i} className="msg-list-item">
              <span className="msg-list-label">{li.label}</span>
              <span className="msg-list-body" dangerouslySetInnerHTML={{ __html: li.body }} />
            </li>
          ))}
        </ol>
      )
      listItems = []
    }
  }

  for (const line of lines) {
    if (!line.trim()) continue

    const listMatch = line.match(/^(\d+)\.\s+\*\*(.+?)\*\*[:]\s*(.*)$/)
    if (listMatch) {
      listItems.push({ label: listMatch[2] + ':', body: listMatch[3] })
      continue
    }

    flushList()

    // Bold inline **text**
    const html = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    elements.push(<p key={key++} className="msg-para" dangerouslySetInnerHTML={{ __html: html }} />)
  }

  flushList()
  return elements
}

export default function ChatMessage({ role, content, streaming = false, citations = [], error = false }) {
  const [citationsOpen, setCitationsOpen] = useState(false)
  const isUser = role === 'user'

  return (
    <div className={`chat-msg ${isUser ? 'chat-msg--user' : 'chat-msg--assistant'}`}>
      {isUser ? (
        <div className="chat-msg-user-row">
          <div className="chat-msg-user-avatar">AN</div>
          <p className="chat-msg-user-text">{content}</p>
          <button className="chat-msg-edit-btn" title="Edit">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <rect x="1.5" y="1.5" width="11" height="11" rx="2" stroke="currentColor" strokeWidth="1.3" />
              <path d="M4.5 9.5l1-3 5-5 2 2-5 5-3 1z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      ) : (
        <div className="chat-msg-assistant">
          <div className="chat-msg-brand">
            <span className="chat-msg-brand-label">CHAT A.I +</span>
            <button className="chat-msg-brand-info" title="About">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.3" />
                <path d="M7 6v4M7 4.5v.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
            </button>
          </div>
          {error ? (
            <p className="chat-msg-error">Something went wrong. Please try again.</p>
          ) : (
            <div className="chat-msg-body">
              {renderContent(content)}
              {streaming && <span className="chat-msg-cursor" aria-hidden="true" />}
            </div>
          )}
          {!streaming && citations.length > 0 && (
            <div className="chat-msg-citations">
              <button
                className="chat-msg-citations-toggle"
                onClick={() => setCitationsOpen((o) => !o)}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ transform: citationsOpen ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>
                  <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {citations.length} source{citations.length !== 1 ? 's' : ''}
              </button>
              {citationsOpen && (
                <div className="chat-msg-citations-list">
                  {citations.map((c) => (
                    <div key={c.chunk_index} className="chat-msg-citation-item">
                      <span className="chat-msg-citation-index">#{c.chunk_index + 1}</span>
                      <p className="chat-msg-citation-text">{c.content}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          <div className="chat-msg-actions">
            <div className="chat-msg-reactions">
              <button className="chat-msg-action-btn" title="Thumbs up">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                  <path d="M5 13V7.5L8 2l1 .5v4h4l-1 6H5zM2 7.5h3V13H2V7.5z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
                </svg>
              </button>
              <span className="chat-msg-action-divider" />
              <button className="chat-msg-action-btn" title="Thumbs down">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                  <path d="M10 2v5.5L7 13l-1-.5v-4H2l1-6h7zM13 7.5h-3V2h3v5.5z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
                </svg>
              </button>
              <span className="chat-msg-action-divider" />
              <button className="chat-msg-action-btn" title="Copy">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                  <rect x="5" y="5" width="8" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.3" />
                  <path d="M3 10H2.5A1.5 1.5 0 011 8.5v-7A1.5 1.5 0 012.5 0h7A1.5 1.5 0 0111 1.5V2" stroke="currentColor" strokeWidth="1.3" />
                </svg>
              </button>
              <span className="chat-msg-action-divider" />
              <button className="chat-msg-action-btn" title="More">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                  <circle cx="3.5" cy="7.5" r="1" fill="currentColor" />
                  <circle cx="7.5" cy="7.5" r="1" fill="currentColor" />
                  <circle cx="11.5" cy="7.5" r="1" fill="currentColor" />
                </svg>
              </button>
            </div>
            <button className="chat-msg-regenerate">
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                <path d="M2 7a5 5 0 109.5-2.2M11.5 2v3h-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Regenerate
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
