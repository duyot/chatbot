import './DocumentSelector.css'

export default function DocumentSelector({ documents = [], value, onChange }) {
  return (
    <div className="doc-selector-row">
      <label className="doc-selector-label" htmlFor="doc-select">
        Chat with:
      </label>
      <select
        id="doc-select"
        className="doc-selector-select"
        value={value ?? ''}
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
