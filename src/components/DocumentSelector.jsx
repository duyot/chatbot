import { useState, useEffect, useCallback } from 'react'
import './DocumentSelector.css'

export default function DocumentSelector({ onChange }) {
  const [documents, setDocuments] = useState([])

  const fetchDocs = useCallback(async () => {
    try {
      const res = await fetch('/api/documents')
      if (res.ok) setDocuments(await res.json())
    } catch (_) {}
  }, [])

  useEffect(() => {
    fetchDocs()
    window.addEventListener('focus', fetchDocs)
    return () => window.removeEventListener('focus', fetchDocs)
  }, [fetchDocs])

  return (
    <div className="doc-selector-row">
      <label className="doc-selector-label" htmlFor="doc-select">
        Chat with:
      </label>
      <select
        id="doc-select"
        className="doc-selector-select"
        defaultValue=""
        onChange={(e) => onChange(e.target.value || null)}
      >
        <option value="">Select a document…</option>
        {documents.map((d) => (
          <option key={d.id} value={d.id}>
            {d.file_name}
          </option>
        ))}
      </select>
    </div>
  )
}
