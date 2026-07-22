import { useEffect, useRef } from 'react'
import ChatMessage from './ChatMessage'
import './ChatThread.css'

export default function ChatThread({ messages, pending, citationEnabled = false, onShowCitation }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, pending])

  return (
    <div className="chat-thread">
      {messages.map((msg) => {
        const citations = msg.citations ?? []
        const canCite =
          citationEnabled &&
          msg.role === 'assistant' &&
          citations.some((c) => Array.isArray(c.bbox) && c.bbox.length > 0)
        return (
          <ChatMessage
            key={msg.id}
            role={msg.role}
            content={msg.content}
            streaming={msg.streaming ?? false}
            citations={citations}
            error={msg.error ?? false}
            showCitation={canCite}
            onShowCitation={() => onShowCitation?.(citations)}
          />
        )
      })}
      <div ref={bottomRef} />
    </div>
  )
}
