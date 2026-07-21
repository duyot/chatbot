import { useState, useEffect } from 'react'
import './ThinkingIndicator.css'

// Claude-Code-style whimsical status words. Cosmetic rotation — edit freely.
const WORDS = ['Processing', 'Ingesting', 'Sprouting', 'Retrieving', 'Reranking', 'Synthesizing']
const ROTATE_MS = 2500
const LONG_WAIT_MS = 12000

export default function ThinkingIndicator() {
  const [wordIndex, setWordIndex] = useState(0)
  const [longWait, setLongWait] = useState(false)

  useEffect(() => {
    const rotate = setInterval(() => {
      setWordIndex((i) => (i + 1) % WORDS.length)
    }, ROTATE_MS)
    const longTimer = setTimeout(() => setLongWait(true), LONG_WAIT_MS)
    return () => {
      clearInterval(rotate)
      clearTimeout(longTimer)
    }
  }, [])

  return (
    <div className="thinking" role="status" aria-live="polite" aria-label="Generating response">
      <div className="thinking-row">
        <span className="thinking-glyph" aria-hidden="true">✳</span>
        <span className="thinking-word" aria-hidden="true">{WORDS[wordIndex]}…</span>
      </div>
      {longWait && (
        <p className="thinking-longwait" aria-hidden="true">
          This is taking longer than expected…
        </p>
      )}
    </div>
  )
}
