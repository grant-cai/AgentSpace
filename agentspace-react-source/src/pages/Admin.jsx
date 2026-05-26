import { useState, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { AGENTS } from '../data/agents'
import AgentIcon from '../components/AgentIcon'
import { useTheme } from '../context/ThemeContext'


const ACTIVE_AGENTS = AGENTS.filter(a => !a.comingSoon)

const API_BASE = 'http://localhost:8000'

function readFileAsBase64(file, onProgress) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result.split(',')[1])
    reader.onprogress = e => {
      if (e.lengthComputable) onProgress?.(Math.round((e.loaded / e.total) * 35))
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

function postIngest(payload, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/ingest`)
    xhr.setRequestHeader('Content-Type', 'application/json')
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress?.(35 + Math.round((e.loaded / e.total) * 60))
    }
    xhr.onload = () => {
      let result
      try { result = JSON.parse(xhr.responseText || '{}') } catch { result = {} }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(result)
      } else {
        reject(new Error(result.detail || xhr.statusText || 'Ingest failed'))
      }
    }
    xhr.onerror = () => reject(new Error('Upload failed'))
    xhr.send(JSON.stringify(payload))
  })
}

const load = (key, fallback) => {
  try { const s = localStorage.getItem(key); return s ? JSON.parse(s) : fallback } catch { return fallback }
}
const save = (key, val) => { try { localStorage.setItem(key, JSON.stringify(val)) } catch { return } }

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(ts) {
  return new Date(ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })
}

export default function Admin() {
  const { C } = useTheme()
  const [selectedId, setSelectedId] = useState(ACTIVE_AGENTS[0]?.id ?? null)
  const [docs, setDocs] = useState(() => load('as_docs', {}))
  const [dragging, setDragging] = useState(false)
  const [toast, setToast] = useState(null)
  const [uploadItems, setUploadItems] = useState([])

  
  const fileRef = useRef(null)

  const [uploading, setUploading] = useState(false)

  const selectedAgent = ACTIVE_AGENTS.find(a => a.id === selectedId)
  const agentDocs = docs[selectedId] || []

  

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  const addDocs = useCallback(async (files) => {
  const pdfs = Array.from(files).filter(f => f.type === 'application/pdf')
  if (pdfs.length === 0) { showToast('Only PDF files are accepted', 'error'); return }

  setUploading(true)
  const batch = pdfs.map(file => ({ id: `${Date.now()}-${file.name}-${Math.random()}`, name: file.name, progress: 0, status: 'Preparing' }))
  setUploadItems(batch)
  const succeeded = []

  const updateUpload = (id, patch) => {
    setUploadItems(items => items.map(item => item.id === id ? { ...item, ...patch } : item))
  }

  for (const [index, file] of pdfs.entries()) {
    const itemId = batch[index].id
    try {
      updateUpload(itemId, { status: 'Reading file', progress: 5 })
      const base64Data = await readFileAsBase64(file, progress => {
        updateUpload(itemId, { progress: Math.max(5, progress) })
      })
      updateUpload(itemId, { status: 'Uploading', progress: 40 })
      await postIngest({ path: file.name, index_name: 'essay_chunk_agentspace', file_data: base64Data }, progress => {
        updateUpload(itemId, { progress: Math.min(95, progress) })
      })
      updateUpload(itemId, { status: 'Complete', progress: 100 })
      succeeded.push(file)
    } catch (err) {
      updateUpload(itemId, { status: 'Failed', progress: 100, error: true })
      showToast(`${file.name}: ${err.message}`, 'error')
    }
  }

  if (succeeded.length > 0) {
    const newDocs = succeeded.map(f => ({ id: Date.now() + Math.random(), name: f.name, size: f.size, uploadedAt: Date.now() }))
    setDocs(prev => {
      const next = { ...prev, [selectedId]: [...(prev[selectedId] || []), ...newDocs] }
      save('as_docs', next)
      return next
    })
    showToast(`${succeeded.length} file${succeeded.length > 1 ? 's' : ''} ingested`)
  }

  setUploading(false)
  setTimeout(() => setUploadItems([]), 1400)
}, [selectedId])

  const deleteDoc = (docId) => {
    setDocs(prev => {
      const next = { ...prev, [selectedId]: prev[selectedId].filter(d => d.id !== docId) }
      save('as_docs', next)
      return next
    })
  }

  const onDrop = e => { e.preventDefault(); setDragging(false); addDocs(e.dataTransfer.files) }
  const onDragOver = e => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const totalDocs = ACTIVE_AGENTS.reduce((sum, a) => sum + (docs[a.id]?.length || 0), 0)

  return (
    <div style={{ display: 'flex', height: '100vh', background: C.bg, color: C.ink, fontFamily: "'DM Sans', sans-serif", overflow: 'hidden' }}>

      {/* Left sidebar */}
      <div style={{ width: 260, flexShrink: 0, background: C.card, borderRight: `1px solid ${C.line}`, display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Sidebar header */}
        <div style={{ padding: '20px 20px 16px', borderBottom: `1px solid ${C.line}` }}>
          <Link to="/" style={{ textDecoration: 'none', color: C.faint, fontSize: 12, fontWeight: 600, letterSpacing: 0.5, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 16 }}>
            ← AgentSpace
          </Link>
          <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, color: C.ink, letterSpacing: '-0.3px' }}>Admin</div>
          <div style={{ fontSize: 12, color: C.faint, marginTop: 4 }}>{totalDocs} document{totalDocs !== 1 ? 's' : ''} uploaded</div>
        </div>

        {/* Agent list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 10px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1.4, color: C.faint, textTransform: 'uppercase', padding: '4px 10px 8px' }}>Agents</div>
          {ACTIVE_AGENTS.map(a => {
            const count = docs[a.id]?.length || 0
            const active = a.id === selectedId
            return (
              <button key={a.id} onClick={() => setSelectedId(a.id)}
                style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 12, padding: '10px 10px', borderRadius: 10, border: 'none', cursor: 'pointer', fontFamily: 'inherit', background: active ? C.subtle : 'transparent', transition: 'background 0.15s', textAlign: 'left' }}
                onMouseEnter={e => { if (!active) e.currentTarget.style.background = C.subtle }}
                onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent' }}>
                <AgentIcon agent={a} size={36} pfx={`adm${a.id}`}/>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: active ? 700 : 500, color: C.ink, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{a.name}</div>
                  <div style={{ fontSize: 11, color: C.faint, marginTop: 1 }}>{a.role}</div>
                </div>
                {count > 0 && (
                  <span style={{ background: a.c1, color: '#fff', fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 10, flexShrink: 0 }}>{count}</span>
                )}
              </button>
            )
          })}
        </div>

      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Top bar */}
        <div style={{ padding: '0 32px', height: 60, borderBottom: `1px solid ${C.line}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: C.card, flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {selectedAgent && <AgentIcon agent={selectedAgent} size={32} pfx="topbar"/>}
            <div>
              <span style={{ fontFamily: "'DM Serif Display', serif", fontSize: 18, color: C.ink }}>{selectedAgent?.name}</span>
              <span style={{ color: C.faint, fontSize: 13, marginLeft: 10 }}>{selectedAgent?.role}</span>
            </div>
          </div>
          <div style={{ fontSize: 13, color: C.faint }}>{agentDocs.length} document{agentDocs.length !== 1 ? 's' : ''}</div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '32px' }}>

          {/* Upload zone */}
          <div
            onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
            onClick={() => fileRef.current.click()}
            style={{ border: `2px dashed ${dragging ? selectedAgent?.c1 || C.accent : C.line}`, borderRadius: 16, padding: '48px 24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, cursor: 'pointer', background: dragging ? `${selectedAgent?.c1}08` : C.card, transition: 'all 0.2s', marginBottom: 32 }}>
            <div style={{ width: 52, height: 52, borderRadius: 14, background: dragging ? `${selectedAgent?.c1}18` : C.subtle, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'background 0.2s' }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={dragging ? selectedAgent?.c1 : C.faint} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>{uploading ? 'Uploading PDFs' : dragging ? 'Drop to upload' : 'Drag & drop PDFs here'}</div>
              <div style={{ fontSize: 13, color: C.faint, marginTop: 4 }}>or click to browse — PDF files only</div>
            </div>
          </div>
          <input ref={fileRef} type="file" accept=".pdf,application/pdf" multiple style={{ display: 'none' }} onChange={e => { addDocs(e.target.files); e.target.value = '' }}/>

          {uploadItems.length > 0 && (
            <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 14, padding: '14px 16px', marginTop: -18, marginBottom: 32, display: 'flex', flexDirection: 'column', gap: 12 }}>
              {uploadItems.map(item => (
                <div key={item.id}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, marginBottom: 8 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: C.ink, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.name}</div>
                    <div style={{ fontSize: 12, color: item.error ? C.accent : C.faint, flexShrink: 0 }}>{item.status} · {item.progress}%</div>
                  </div>
                  <div style={{ height: 8, borderRadius: 999, background: C.subtle, overflow: 'hidden' }}>
                    <div style={{ width: `${item.progress}%`, height: '100%', borderRadius: 999, background: item.error ? C.accent : selectedAgent?.c1 || C.accent, transition: 'width 0.2s ease' }}/>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Document log */}
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, color: C.faint, textTransform: 'uppercase', marginBottom: 14 }}>
            Document log
          </div>

          {agentDocs.length === 0 ? (
            <div style={{ background: C.card, borderRadius: 14, border: `1px solid ${C.line}`, padding: '40px 24px', textAlign: 'center' }}>
              <div style={{ fontSize: 14, color: C.faint }}>No documents uploaded for {selectedAgent?.name} yet</div>
            </div>
          ) : (
            <div style={{ background: C.card, borderRadius: 14, border: `1px solid ${C.line}`, overflow: 'hidden' }}>
              {/* Table header */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 100px 180px 40px', gap: '0 16px', padding: '10px 20px', borderBottom: `1px solid ${C.line}`, background: C.subtle }}>
                {['Filename', 'Size', 'Uploaded', ''].map(h => (
                  <div key={h} style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.8, color: C.faint, textTransform: 'uppercase' }}>{h}</div>
                ))}
              </div>
              {/* Rows */}
              {[...agentDocs].reverse().map((doc, i) => (
                <div key={doc.id}
                  style={{ display: 'grid', gridTemplateColumns: '1fr 100px 180px 40px', gap: '0 16px', padding: '14px 20px', borderBottom: i < agentDocs.length - 1 ? `1px solid ${C.line}` : 'none', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    <span style={{ fontSize: 14, color: C.ink, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{doc.name}</span>
                  </div>
                  <div style={{ fontSize: 13, color: C.muted }}>{formatSize(doc.size)}</div>
                  <div style={{ fontSize: 13, color: C.muted }}>{formatDate(doc.uploadedAt)}</div>
                  <button onClick={() => deleteDoc(doc.id)}
                    style={{ width: 28, height: 28, borderRadius: 6, border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.faint, fontSize: 16, transition: 'color 0.15s, background 0.15s' }}
                    onMouseEnter={e => { e.currentTarget.style.color = C.accent; e.currentTarget.style.background = C.subtle }}
                    onMouseLeave={e => { e.currentTarget.style.color = C.faint; e.currentTarget.style.background = 'transparent' }}>×</button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div style={{ position: 'fixed', bottom: 24, right: 24, padding: '12px 20px', borderRadius: 12, background: toast.type === 'error' ? '#fee2e2' : '#dcfce7', color: toast.type === 'error' ? '#b91c1c' : '#15803d', fontSize: 14, fontWeight: 600, boxShadow: '0 4px 20px rgba(0,0,0,0.12)', animation: 'fadein 0.2s ease', zIndex: 999 }}>
          {toast.type === 'error' ? '✕ ' : '✓ '}{toast.msg}
        </div>
      )}
    </div>
  )
}
