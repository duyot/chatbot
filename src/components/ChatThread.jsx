import { useEffect, useRef } from 'react'
import ChatMessage from './ChatMessage'
import './ChatThread.css'

export default function ChatThread({ messages, pending }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, pending])

  const anyStreaming = messages.some((m) => m.streaming)

  return (
    <div className="chat-thread">
      {messages.map((msg) => (
        <ChatMessage
          key={msg.id}
          role={msg.role}
          content={msg.content}
          streaming={msg.streaming ?? false}
          citations={msg.citations ?? []}
          error={msg.error ?? false}
        />
      ))}
      {pending && !anyStreaming && (
        <div className="chat-thread-typing">
          <span className="chat-thread-typing-label">CHAT A.I +</span>
          <span className="chat-thread-typing-dots">
            <span /><span /><span />
          </span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
