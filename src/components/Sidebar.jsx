import { useState } from 'react'
import './Sidebar.css'

const CONVERSATIONS = [
  { id: 'c1', title: 'Create Html Game Environment...' },
  { id: 'c2', title: 'Apply To Leave For Emergency' },
  { id: 'c3', title: 'What Is UI UX Design?' },
  { id: 'c4', title: 'Create POS System' },
  { id: 'c5', title: 'What Is UX Audit?' },
  { id: 'c6', title: 'Create Chatbot GPT...' },
  { id: 'c7', title: 'How Chat GPT Work?' },
]

const LAST_7_DAYS = [
  { id: 'c8', title: 'Crypto Lending App Name' },
  { id: 'c9', title: 'Operator Grammar Types' },
  { id: 'c10', title: 'Min States For Binary DFA', disabled: true },
]

function ChatIcon({ active }) {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" className="conv-icon">
      <path
        d="M1.5 2.5h12a.5.5 0 01.5.5v7a.5.5 0 01-.5.5H9l-1.5 2-1.5-2H1.5A.5.5 0 011 10V3a.5.5 0 01.5-.5z"
        stroke={active ? '#0071e3' : 'currentColor'}
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export default function Sidebar({ activeTitle }) {
  const [activeId, setActiveId] = useState(
    CONVERSATIONS.find(c => c.title.startsWith(activeTitle?.slice(0, 10)))?.id || 'c6'
  )
  const [hoveredId, setHoveredId] = useState(null)

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <span className="sidebar-logo-text">CHAT A.I+</span>
      </div>

      {/* New chat + Search */}
      <div className="sidebar-actions">
        <button className="sidebar-new-chat">
          <span className="sidebar-new-chat-plus">+</span>
          New chat
        </button>
        <button className="sidebar-search-btn" title="Search">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="7" cy="7" r="4.5" stroke="white" strokeWidth="1.6" />
            <path d="M10.5 10.5L13 13" stroke="white" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      {/* Conversations list */}
      <div className="sidebar-section-header">
        <span>Your conversations</span>
        <button className="sidebar-clear-all">Clear All</button>
      </div>

      <nav className="sidebar-nav">
        {CONVERSATIONS.map(item => {
          const isActive = item.id === activeId
          const isHovered = item.id === hoveredId

          return (
            <button
              key={item.id}
              className={`sidebar-item ${isActive ? 'is-active' : ''}`}
              onClick={() => setActiveId(item.id)}
              onMouseEnter={() => setHoveredId(item.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              <ChatIcon active={isActive} />
              <span className="sidebar-item-title">{item.title}</span>
              {(isActive || isHovered) && (
                <span className="sidebar-item-actions" onClick={e => e.stopPropagation()}>
                  <button className="sidebar-action-btn" title="Delete">
                    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                      <path d="M2 3.5h9M4.5 3.5V2.5h4v1M5.5 6v3.5M7.5 6v3.5M3 3.5l.5 7h6l.5-7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                  <button className="sidebar-action-btn" title="Edit">
                    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                      <path d="M9 2l2 2-7 7H2V9L9 2z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
                    </svg>
                  </button>
                  {isActive && <span className="sidebar-active-dot" />}
                </span>
              )}
            </button>
          )
        })}

        <div className="sidebar-group-label">Last 7 Days</div>

        {LAST_7_DAYS.map(item => (
          <button
            key={item.id}
            className={`sidebar-item ${item.disabled ? 'is-disabled' : ''} ${item.id === activeId ? 'is-active' : ''}`}
            onClick={() => !item.disabled && setActiveId(item.id)}
            disabled={item.disabled}
          >
            <ChatIcon active={item.id === activeId} />
            <span className="sidebar-item-title">{item.title}</span>
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <button className="sidebar-footer-item">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="2.5" stroke="currentColor" strokeWidth="1.4" />
            <path d="M8 1.5v1.2M8 13.3v1.2M1.5 8h1.2M13.3 8h1.2M3.4 3.4l.85.85M11.75 11.75l.85.85M3.4 12.6l.85-.85M11.75 4.25l.85-.85" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
          <span>Settings</span>
        </button>
        <div className="sidebar-user">
          <div className="sidebar-user-avatar">AN</div>
          <span className="sidebar-user-name">Andrew Neilson</span>
        </div>
      </div>
    </aside>
  )
}
