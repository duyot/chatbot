import { useEffect, useRef } from 'react'
import { usePageImages } from '../hooks/usePageImages'
import CitationOverlay from './CitationOverlay'
import './PageImageViewer.css'

// How far outside the viewport a page starts loading. Generous enough that a
// normal scroll never waits on a fetch.
const PRELOAD_MARGIN = '600px 0px'

/**
 * Renders a document as a vertical run of server-rendered page images, with
 * citation boxes overlaid on the highlighted page.
 *
 * Each page's wrapper gets its aspect ratio from the manifest, so the scroll
 * height is correct before any image bytes arrive — that's what makes
 * scroll-to-citation land on the right page even for a page far down the
 * document. Citation rects are normalized to the source page rect, and a
 * rendered image is geometrically similar to it, so the rects map on directly.
 *
 * @param {{
 *   documentId: string,
 *   highlightTarget: ({ page: number, rects: number[][] }|null),
 * }} props
 */
export default function PageImageViewer({ documentId, highlightTarget }) {
  const { pages, urls, loading, error, requestPage } = usePageImages(documentId)
  const containerRef = useRef(null)
  const pageRefs = useRef({})

  // Load pages as they approach the viewport. Page 1 is covered too, since it
  // is intersecting from the start.
  useEffect(() => {
    if (pages.length === 0) return undefined

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return
          const page = Number(entry.target.dataset.page)
          if (page) requestPage(page)
        })
      },
      { root: containerRef.current, rootMargin: PRELOAD_MARGIN },
    )

    Object.values(pageRefs.current).forEach((el) => {
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [pages, requestPage])

  // Jump to the cited page, fetching it immediately rather than waiting for the
  // scroll to bring it into the observer's range.
  useEffect(() => {
    if (!highlightTarget || pages.length === 0) return
    requestPage(highlightTarget.page)
    pageRefs.current[highlightTarget.page]?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    })
  }, [highlightTarget, pages, requestPage])

  if (loading) return <p className="file-preview-hint">Loading preview…</p>
  if (error) return <p className="file-preview-hint">Couldn&apos;t load this preview.</p>
  if (pages.length === 0) {
    return <p className="file-preview-hint">No preview is available for this document.</p>
  }

  return (
    <div className="page-image-viewer" ref={containerRef}>
      {pages.map(({ page, width, height }) => {
        const rects = highlightTarget?.page === page ? highlightTarget.rects || [] : []
        const url = urls[page]
        return (
          <div
            key={page}
            className="page-image"
            data-page={page}
            ref={(el) => { pageRefs.current[page] = el }}
            style={{ aspectRatio: `${width} / ${height}` }}
          >
            {url && <img src={url} alt={`Page ${page}`} draggable="false" />}
            <CitationOverlay rects={rects} />
          </div>
        )
      })}
    </div>
  )
}
