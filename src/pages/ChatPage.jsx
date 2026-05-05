import { useState, useCallback } from 'react'
import Sidebar from '../components/Sidebar'
import ChatThread from '../components/ChatThread'
import Composer from '../components/Composer'
import UpgradeTab from '../components/UpgradeTab'
import UploadToast from '../components/UploadToast'
import DocumentSelector from '../components/DocumentSelector'
import { useUpload } from '../hooks/useUpload'
import { useChat } from '../hooks/useChat'
import './ChatPage.css'

export default function ChatPage() {
  const { messages, pending, handleSend } = useChat()
  const [selectedDocumentId, setSelectedDocumentId] = useState(null)
  const [showToast, setShowToast] = useState(false)

  const upload = useUpload({
    onComplete: () => setShowToast(true),
  })

  const onSend = useCallback((text) => {
    handleSend(text, selectedDocumentId)
  }, [handleSend, selectedDocumentId])

  return (
    <div className="chat-page-outer">
      <div className="chat-page-card">
        <Sidebar activeTitle="Chat with Document" />
        <div className="chat-main">
          <DocumentSelector onChange={setSelectedDocumentId} />
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
        <div className="chat-page-upgrade">
          <UpgradeTab />
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
