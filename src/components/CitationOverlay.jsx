import './CitationOverlay.css'

/**
 * Draws citation highlight boxes over a positioned parent (e.g. the wrapper
 * around a document image). Each rect is normalized [x0,y0,x1,y1] in [0,1] of
 * the page, so boxes scale automatically with the rendered image size.
 *
 * @param {{ rects: number[][] }} props
 */
export default function CitationOverlay({ rects }) {
  if (!rects || rects.length === 0) return null
  return (
    <div className="citation-overlay" aria-hidden="true">
      {rects.map(([x0, y0, x1, y1], i) => (
        <div
          key={i}
          className="citation-box"
          style={{
            left: `${x0 * 100}%`,
            top: `${y0 * 100}%`,
            width: `${(x1 - x0) * 100}%`,
            height: `${(y1 - y0) * 100}%`,
          }}
        />
      ))}
    </div>
  )
}
