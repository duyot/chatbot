import { useState, useRef } from 'react'

export function useUpload({ onComplete } = {}) {
  const [state, setState] = useState({
    status: 'idle',   // idle | uploading | processing | done | failed
    fileName: null,
    error: null,
  })
  const eventSourceRef = useRef(null)

  async function uploadFile(file) {
    setState({ status: 'uploading', fileName: file.name, error: null })

    const formData = new FormData()
    formData.append('file', file)

    let docId
    try {
      const res = await fetch('/api/documents/upload', { method: 'POST', body: formData })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      docId = data.id
    } catch (err) {
      setState({ status: 'failed', fileName: file.name, error: err.message })
      return
    }

    setState(s => ({ ...s, status: 'processing' }))

    const es = new EventSource(`/api/documents/${docId}/status`)
    eventSourceRef.current = es

    es.onmessage = (e) => {
      const payload = JSON.parse(e.data)
      if (payload.status === 'done') {
        es.close()
        setState({ status: 'done', fileName: file.name, error: null })
        onComplete?.(file.name)
      } else if (payload.status === 'failed') {
        es.close()
        setState({ status: 'failed', fileName: file.name, error: payload.message })
      }
    }

    es.onerror = () => {
      es.close()
      setState(s => ({ ...s, status: 'failed', error: 'Connection to server lost.' }))
    }
  }

  function reset() {
    eventSourceRef.current?.close()
    setState({ status: 'idle', fileName: null, error: null })
  }

  return { ...state, uploadFile, reset }
}
