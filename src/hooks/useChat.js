import { useReducer, useCallback, useRef } from 'react'

let nextId = 1
function makeId() { return nextId++ }

function reducer(state, action) {
  switch (action.type) {
    case 'ADD_USER_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, { id: action.id, role: 'user', content: action.content }],
        pending: true,
      }
    case 'START_ASSISTANT_MESSAGE':
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: action.id, role: 'assistant', content: '', streaming: true, citations: [], error: false },
        ],
      }
    case 'APPEND_TOKEN': {
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, content: m.content + action.content } : m
        ),
      }
    }
    case 'SET_CITATIONS':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, citations: action.chunks } : m
        ),
      }
    case 'END_STREAMING':
      return {
        ...state,
        pending: false,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, streaming: false } : m
        ),
      }
    case 'SET_ERROR':
      return {
        ...state,
        pending: false,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, streaming: false, error: true } : m
        ),
      }
    default:
      return state
  }
}

export function useChat() {
  const [state, dispatch] = useReducer(reducer, { messages: [], pending: false })
  const abortRef = useRef(null)

  const handleSend = useCallback(async (message, documentId) => {
    if (!documentId || !message.trim()) return

    dispatch({ type: 'ADD_USER_MESSAGE', id: makeId(), content: message })
    const assistantId = makeId()
    dispatch({ type: 'START_ASSISTANT_MESSAGE', id: assistantId })

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: documentId, message }),
        signal: controller.signal,
      })

      if (!res.ok) {
        dispatch({ type: 'SET_ERROR', id: assistantId })
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw) continue
          let event
          try { event = JSON.parse(raw) } catch { continue }
          if (event.type === 'token') {
            dispatch({ type: 'APPEND_TOKEN', id: assistantId, content: event.content })
          } else if (event.type === 'citations') {
            dispatch({ type: 'SET_CITATIONS', id: assistantId, chunks: event.chunks })
          } else if (event.type === 'done') {
            dispatch({ type: 'END_STREAMING', id: assistantId })
          } else if (event.type === 'error') {
            dispatch({ type: 'SET_ERROR', id: assistantId })
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        dispatch({ type: 'SET_ERROR', id: assistantId })
      }
    }
  }, [])

  return { messages: state.messages, pending: state.pending, handleSend }
}
