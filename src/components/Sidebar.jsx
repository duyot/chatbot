import { useState } from 'react'
import { useAuth } from '../auth/AuthContext'
import './Sidebar.css'

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

function initials(name) {
  if (!name) return '?'
  const parts = name.trim().split(/\s+/)
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase()
  return name.slice(0, 2).toUpperCase()
}

export default function Sidebar({
  conversations = [],
  activeId,
  onSelect,
  onNewChat,
  onDelete,
  onClearAll,
  onToggle,
}) {
  const { user, logout } = useAuth()
  const [hoveredId, setHoveredId] = useState(null)
  const userName = user?.username || 'User'

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <span className="sidebar-logo-text">CHAT A.I+</span>
        <button className="sidebar-collapse-btn" title="Collapse sidebar" onClick={onToggle}>
          <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
            <rect x="1.5" y="2" width="12" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
            <path d="M5.5 2v11" stroke="currentColor" strokeWidth="1.2" />
          </svg>
        </button>
      </div>

      {/* New chat + Search */}
      <div className="sidebar-actions">
        <button className="sidebar-new-chat" onClick={onNewChat}>
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
        {conversations.length > 0 && (
          <button className="sidebar-clear-all" onClick={onClearAll}>Clear All</button>
        )}
      </div>

      <nav className="sidebar-nav">
        {conversations.length === 0 && (
          <p style={{ padding: '0 18px', color: '#9aa0a6', fontSize: 13 }}>
            No conversations yet.
          </p>
        )}

        {conversations.map((item) => {
          const isActive = item.id === activeId
          const isHovered = item.id === hoveredId

          return (
            <button
              key={item.id}
              className={`sidebar-item ${isActive ? 'is-active' : ''}`}
              onClick={() => onSelect?.(item.id)}
              onMouseEnter={() => setHoveredId(item.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              <ChatIcon active={isActive} />
              <span className="sidebar-item-title">{item.title}</span>
              {(isActive || isHovered) && (
                <span className="sidebar-item-actions" onClick={(e) => e.stopPropagation()}>
                  <button
                    className="sidebar-action-btn"
                    title="Delete"
                    onClick={() => onDelete?.(item.id)}
                  >
                    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                      <path d="M2 3.5h9M4.5 3.5V2.5h4v1M5.5 6v3.5M7.5 6v3.5M3 3.5l.5 7h6l.5-7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                </span>
              )}
            </button>
          )
        })}
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
        <button className="sidebar-footer-item sidebar-footer-item--logout" onClick={logout}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M6 2.5H3.5a1 1 0 00-1 1v9a1 1 0 001 1H6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M10 11l3-3-3-3M13 8H6.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span>Log out</span>
        </button>
        <div className="sidebar-user">
          <div className="sidebar-user-avatar">{initials(userName)}</div>
          <span className="sidebar-user-name">{userName}</span>
        </div>
      </div>
    </aside>
  )
}
