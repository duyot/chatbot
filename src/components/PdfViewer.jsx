import { useEffect, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import CitationOverlay from './CitationOverlay'
import './PdfViewer.css'

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl

const SCALE = 1.5

/**
 * Renders every page of a PDF to canvas and overlays citation boxes on the
 * highlighted page. Page geometry (from PyMuPDF) and pdf.js viewports both use a
 * top-left origin, so normalized rects map directly onto the rendered page.
 *
 * @param {{ blobUrl: string, highlightTarget: ({ page: number, rects: number[][] }|null) }} props
 */
export default function PdfViewer({ blobUrl, highlightTarget }) {
  const [pages, setPages] = useState([])   // [{ n, w, h }]
  const [error, setError] = useState(false)
  const pdfRef = useRef(null)
  const canvasRefs = useRef({})
  const pageRefs = useRef({})

  // Load the document and measure each page's viewport.
  useEffect(() => {
    if (!blobUrl) return undefined
    let cancelled = false
    canvasRefs.current = {}
    pageRefs.current = {}

    async function load() {
      setPages([])
      setError(false)
      try {
        const pdf = await pdfjsLib.getDocument(blobUrl).promise
        if (cancelled) return
        pdfRef.current = pdf
        const meta = []
        for (let n = 1; n <= pdf.numPages; n++) {
          const page = await pdf.getPage(n)
          if (cancelled) return
          const vp = page.getViewport({ scale: SCALE })
          meta.push({ n, w: Math.floor(vp.width), h: Math.floor(vp.height) })
        }
        if (!cancelled) setPages(meta)
      } catch {
        if (!cancelled) setError(true)
      }
    }

    load()
    return () => {
      cancelled = true
      pdfRef.current?.destroy?.()
      pdfRef.current = null
    }
  }, [blobUrl])

  // Paint each page into its canvas once the placeholders exist in the DOM.
  useEffect(() => {
    const pdf = pdfRef.current
    if (!pdf || pages.length === 0) return undefined
    let cancelled = false
    const tasks = []

    async function render() {
      for (const { n } of pages) {
        const canvas = canvasRefs.current[n]
        if (!canvas) continue
        const page = await pdf.getPage(n)
        if (cancelled) return
        const vp = page.getViewport({ scale: SCALE })
        canvas.width = vp.width
        canvas.height = vp.height
        const task = page.render({ canvasContext: canvas.getContext('2d'), viewport: vp })
        tasks.push(task)
        try {
          await task.promise
        } catch {
          /* render cancelled on unmount / reload */
        }
      }
    }

    render()
    return () => {
      cancelled = true
      tasks.forEach((t) => t.cancel?.())
    }
  }, [pages])

  // Scroll to the highlighted page whenever the target changes.
  useEffect(() => {
    if (!highlightTarget || pages.length === 0) return
    pageRefs.current[highlightTarget.page]?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [highlightTarget, pages])

  if (error) return <p className="file-preview-hint">Couldn&apos;t render this PDF.</p>

  return (
    <div className="pdf-viewer">
      {pages.map(({ n, w, h }) => {
        const rects = highlightTarget && highlightTarget.page === n ? highlightTarget.rects || [] : []
        return (
          <div
            key={n}
            className="pdf-page"
            ref={(el) => { pageRefs.current[n] = el }}
            style={{ width: w, aspectRatio: `${w} / ${h}` }}
          >
            <canvas ref={(el) => { canvasRefs.current[n] = el }} />
            <CitationOverlay rects={rects} />
          </div>
        )
      })}
    </div>
  )
}
