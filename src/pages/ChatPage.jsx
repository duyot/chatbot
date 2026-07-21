import { useState, useCallback } from 'react'
import Sidebar from '../components/Sidebar'
import ChatThread from '../components/ChatThread'
import Composer from '../components/Composer'
import UploadToast from '../components/UploadToast'
import DocumentSelector from '../components/DocumentSelector'
import FilePreview from '../components/FilePreview'
import { useUpload } from '../hooks/useUpload'
import { useChat } from '../hooks/useChat'
import { useConversations } from '../hooks/useConversations'
import { useDocuments } from '../hooks/useDocuments'
import { getConversation } from '../repositories/conversationRepository'
import './ChatPage.css'

const SIDEBAR_OPEN_KEY = 'chatbot_sidebar_open'

function ExpandIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <rect x="1.5" y="2" width="12" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M5.5 2v11" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  )
}

export default function ChatPage() {
  const { messages, pending, handleSend, loadMessages, reset } = useChat()
  const { conversations, refresh, remove, clearAll } = useConversations()
  const { documents } = useDocuments()
  const [selectedDocumentId, setSelectedDocumentId] = useState(null)
  const [activeConversationId, setActiveConversationId] = useState(null)
  const [showToast, setShowToast] = useState(false)
  const [sidebarOpen, setSidebarOpenState] = useState(() => {
    const saved = localStorage.getItem(SIDEBAR_OPEN_KEY)
    return saved === null ? true : saved === 'true'
  })

  const setSidebarOpen = useCallback((open) => {
    setSidebarOpenState(open)
    localStorage.setItem(SIDEBAR_OPEN_KEY, String(open))
  }, [])

  const selectedDocument = documents.find((d) => d.id === selectedDocumentId) ?? null

  const upload = useUpload({
    onComplete: () => setShowToast(true),
  })

  // Fired when the stream reports the conversation it created/attached to.
  const handleConversation = useCallback((id) => {
    setActiveConversationId(id)
    refresh()
  }, [refresh])

  const onSend = useCallback((text) => {
    handleSend(text, selectedDocumentId, activeConversationId, handleConversation)
  }, [handleSend, selectedDocumentId, activeConversationId, handleConversation])

  const onNewChat = useCallback(() => {
    reset()
    setActiveConversationId(null)
  }, [reset])

  const onSelectConversation = useCallback(async (id) => {
    setActiveConversationId(id)
    try {
      const conv = await getConversation(id)
      loadMessages(conv.messages)
      // Best-effort: continue against the last document used in this thread.
      const lastWithDoc = [...conv.messages].reverse().find((m) => m.document_id)
      if (lastWithDoc) setSelectedDocumentId(lastWithDoc.document_id)
    } catch {
      /* ignore load errors */
    }
  }, [loadMessages])

  const onDeleteConversation = useCallback(async (id) => {
    try {
      await remove(id)
      if (id === activeConversationId) {
        reset()
        setActiveConversationId(null)
      }
    } catch {
      /* ignore delete errors */
    }
  }, [remove, activeConversationId, reset])

  const onClearAll = useCallback(async () => {
    try {
      await clearAll()
      reset()
      setActiveConversationId(null)
    } catch {
      /* ignore clear errors */
    }
  }, [clearAll, reset])

  return (
    <div className="chat-page-outer">
      <div className="chat-page-card">
        {sidebarOpen ? (
          <Sidebar
            conversations={conversations}
            activeId={activeConversationId}
            onSelect={onSelectConversation}
            onNewChat={onNewChat}
            onDelete={onDeleteConversation}
            onClearAll={onClearAll}
            onToggle={() => setSidebarOpen(false)}
          />
        ) : (
          <button
            className="sidebar-reopen-btn"
            title="Expand sidebar"
            onClick={() => setSidebarOpen(true)}
          >
            <ExpandIcon />
          </button>
        )}
        <div className={`chat-split${selectedDocument ? ' is-split' : ''}`}>
          {selectedDocument && (
            <FilePreview
              document={selectedDocument}
              onClose={() => setSelectedDocumentId(null)}
            />
          )}
          <div className="chat-main">
            <DocumentSelector
              documents={documents}
              value={selectedDocumentId}
              onChange={setSelectedDocumentId}
            />
            {messages.length === 0 && (
              <p className="chat-page-hint">
                {selectedDocumentId
                  ? 'Ask a question about the selected document.'
                  : 'Select a document above to start chatting.'}
              </p>
            )}
            <ChatThread messages={messages} pending={pending} />
            <Composer
              onSend={onSend}
              onFileSelect={upload.uploadFile}
              uploadStatus={upload.status}
              disabled={pending || !selectedDocumentId}
            />
          </div>
        </div>
      </div>
      {showToast && (
        <UploadToast
          status={upload.status}
          fileName={upload.fileName}
          error={upload.error}
          onDismiss={() => {
            setShowToast(false)
            upload.reset()
          }}
        />
      )}
    </div>
  )
}
