import { useState, useEffect, useRef, useCallback } from 'react'
import { AGENTS } from '../data/agents'
import TopBar from '../components/TopBar'
import BrowseScreen from '../components/BrowseScreen'
import DetailScreen from '../components/DetailScreen'
import ChatScreen from '../components/ChatScreen'
import MyAgentsScreen from '../components/MyAgentsScreen'
import SubscribeModal from '../components/SubscribeModal'
import BottomNav from '../components/BottomNav'
import { useTheme } from '../context/ThemeContext'
import '../styles/app.css'

const CHAT_DARK = {
  bg:     '#1a1a1a',  // main chat canvas
  card:   '#242424',  // header + input bar
  sidebar:'#111111',  // left panel — noticeably darker
  ink:    '#f0ede8',  // primary text — warm white, not harsh
  mid:    '#b8b4ae',  // message body text
  muted:  '#7a7672',  // placeholders, tertiary labels
  faint:  '#4a4745',  // very muted — dividers, dates
  subtle: '#2c2c2c',  // hover states
  line:   '#2f2f2f',  // borders
  accent: '#e63c2f',
  green:  '#22c55e',
}

function SessionRow({ s, active, C, onSelect, onDelete, onRename }) {
  const [hov, setHov] = useState(false)
  const [confirm, setConfirm] = useState(false)
  const [ctx, setCtx] = useState(null)       // { x, y }
  const [editing, setEditing] = useState(false)
  const [renameVal, setRenameVal] = useState(s.title)
  const renameRef = useRef(null)

  const openCtx = e => { e.preventDefault(); setCtx({ x: e.clientX, y: e.clientY }) }
  const closeCtx = () => setCtx(null)

  const startRename = () => {
    closeCtx()
    setRenameVal(s.title)
    setEditing(true)
    setTimeout(() => { renameRef.current?.select() }, 0)
  }

  const commitRename = () => {
    const v = renameVal.trim()
    if (v) onRename(s.id, v)
    setEditing(false)
  }

  return (
    <>
      <div onMouseEnter={() => setHov(true)} onMouseLeave={() => { setHov(false); setConfirm(false) }}
        onContextMenu={openCtx}
        style={{ display: 'flex', alignItems: 'center', borderRadius: 8, background: active || hov ? C.subtle : 'transparent', transition: 'background 0.15s', cursor: 'pointer' }}>
        <div onClick={onSelect} style={{ flex: 1, padding: '10px 12px', minWidth: 0 }}>
          {editing ? (
            <input
              ref={renameRef}
              value={renameVal}
              onChange={e => setRenameVal(e.target.value)}
              onBlur={commitRename}
              onKeyDown={e => { if (e.key === 'Enter') commitRename(); if (e.key === 'Escape') setEditing(false) }}
              onClick={e => e.stopPropagation()}
              style={{ width: '100%', fontSize: 13, fontWeight: active ? 600 : 400, color: C.ink, background: 'transparent', border: 'none', borderBottom: `1px solid ${C.line}`, outline: 'none', fontFamily: 'inherit', padding: '1px 0' }}
            />
          ) : (
            <div style={{ fontSize: 13, fontWeight: active ? 600 : 400, color: C.ink, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.title}</div>
          )}
          <div style={{ fontSize: 11, color: C.faint, marginTop: 2 }}>{new Date(s.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</div>
        </div>
        {confirm ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginRight: 6, flexShrink: 0 }}>
            <span style={{ fontSize: 11, color: C.muted }}>Delete?</span>
            <button onClick={e => { e.stopPropagation(); onDelete() }}
              style={{ padding: '2px 7px', borderRadius: 5, border: 'none', background: '#e63c2f', color: '#fff', cursor: 'pointer', fontSize: 11, fontWeight: 600, fontFamily: 'inherit' }}>Yes</button>
            <button onClick={e => { e.stopPropagation(); setConfirm(false) }}
              style={{ padding: '2px 7px', borderRadius: 5, border: `1px solid ${C.line}`, background: 'transparent', color: C.muted, cursor: 'pointer', fontSize: 11, fontWeight: 600, fontFamily: 'inherit' }}>No</button>
          </div>
        ) : (
          <button onClick={e => { e.stopPropagation(); setConfirm(true) }}
            style={{ flexShrink: 0, width: 26, height: 26, borderRadius: 6, border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.faint, fontSize: 15, marginRight: 6, opacity: hov ? 1 : 0, transition: 'opacity 0.15s, color 0.15s' }}
            onMouseEnter={e => e.currentTarget.style.color = '#e63c2f'}
            onMouseLeave={e => e.currentTarget.style.color = C.faint}>×</button>
        )}
      </div>

      {ctx && (
        <>
          <div onClick={closeCtx} style={{ position: 'fixed', inset: 0, zIndex: 200 }}/>
          <div style={{ position: 'fixed', top: ctx.y, left: ctx.x, zIndex: 201, background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, boxShadow: '0 4px 16px rgba(0,0,0,0.15)', padding: '4px', minWidth: 130 }}>
            <button onClick={startRename}
              style={{ display: 'block', width: '100%', padding: '8px 12px', background: 'transparent', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 13, color: C.ink, fontFamily: 'inherit', textAlign: 'left', transition: 'background 0.12s' }}
              onMouseEnter={e => e.currentTarget.style.background = C.subtle}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
              ✎ Rename
            </button>
          </div>
        </>
      )}
    </>
  )
}

const load = (key, fallback) => {
  try { const s = localStorage.getItem(key); return s ? JSON.parse(s) : fallback } catch { return fallback }
}

export default function Browse() {
  const [tab, setTab] = useState('browse')
  const [detailAgent, setDetailAgent] = useState(null)
  const [chatAgent, setChatAgent] = useState(null)
  const [owned, setOwned] = useState(() => load('as_owned', []))
  const [subscribeModal, setSubscribeModal] = useState(false)
  const [vw, setVw] = useState(window.innerWidth)

  // Sessions: [{ id, agentId, title, createdAt }]
  // Messages: { [sessionId]: [{ role, text }] }
  const [sessions, setSessions] = useState(() => load('as_sessions', []))
  const [messages, setMessages] = useState(() => load('as_messages', {}))
  const [activeSessionId, setActiveSessionId] = useState(null)

  useEffect(() => {
    const onResize = () => setVw(window.innerWidth)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  useEffect(() => { try { localStorage.setItem('as_owned',    JSON.stringify(owned))    } catch {} }, [owned])
  useEffect(() => { try { localStorage.setItem('as_sessions', JSON.stringify(sessions))  } catch {} }, [sessions])
  useEffect(() => { try { localStorage.setItem('as_messages', JSON.stringify(messages))  } catch {} }, [messages])

  useEffect(() => {
    document.body.style.overflow = (detailAgent || chatAgent) ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [detailAgent, chatAgent])

  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') { closeDetail(); closeChat() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const { C } = useTheme()
  const isDesktop = vw >= 768
  const [sidebarWidth, setSidebarWidth] = useState(260)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [chatDark, setChatDark] = useState(false)
  const CC = chatDark ? CHAT_DARK : C
  const dragRef = useRef(null)

  const onDragStart = useCallback(e => {
    e.preventDefault()
    const startX = e.clientX
    const startW = sidebarWidth
    const onMove = ev => setSidebarWidth(Math.min(480, Math.max(160, startW + ev.clientX - startX)))
    const onUp   = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [sidebarWidth])

  // ── Detail ──────────────────────────────────────────
  const openDetail = a => setDetailAgent(a)
  const closeDetail = () => { setDetailAgent(null); setSubscribeModal(false) }

  // ── Subscribe ────────────────────────────────────────
  const subscribe   = () => { setOwned(o => [...o, detailAgent.id]); setSubscribeModal(false) }
  const unsubscribe = id => setOwned(o => o.filter(i => i !== id))
  const unsubscribeFromChat = () => {
    if (!chatAgent) return
    unsubscribe(chatAgent.id)
    closeChat()
    setTab('myagents')
  }

  // ── Chat sessions ────────────────────────────────────
  const createSession = agent => {
    const id = Date.now().toString()
    const session = { id, agentId: agent.id, title: 'New conversation', createdAt: Date.now() }
    setSessions(prev => [session, ...prev])
    setMessages(prev => ({ ...prev, [id]: [] }))
    return id
  }

  const openChat = agent => {
    setDetailAgent(null)
    setChatAgent(agent)
    const agentSessions = sessions.filter(s => s.agentId === agent.id)
    if (agentSessions.length === 0) {
      setActiveSessionId(createSession(agent))
    } else {
      setActiveSessionId(agentSessions[0].id)
    }
  }

  const closeChat = () => { setChatAgent(null); setActiveSessionId(null) }

  const newChat = () => setActiveSessionId(createSession(chatAgent))

  const deleteSession = id => {
    setSessions(prev => prev.filter(s => s.id !== id))
    setMessages(prev => { const next = { ...prev }; delete next[id]; return next })
    if (activeSessionId === id) {
      const remaining = sessions.filter(s => s.agentId === chatAgent?.id && s.id !== id)
      setActiveSessionId(remaining.length > 0 ? remaining[0].id : null)
    }
  }

  const selectSession = id => setActiveSessionId(id)
  const renameSession = (id, title) => setSessions(prev => prev.map(s => s.id === id ? { ...s, title } : s))

  const addMsg = (sessionId, msg) => {
    setMessages(prev => ({ ...prev, [sessionId]: [...(prev[sessionId] || []), msg] }))
    if (msg.role === 'user') {
      setSessions(prev => prev.map(s =>
        s.id === sessionId && s.title === 'New conversation'
          ? { ...s, title: msg.text.slice(0, 45) }
          : s
      ))
    }
  }

  const agentSessions = chatAgent ? sessions.filter(s => s.agentId === chatAgent.id) : []
  const activeHistory = activeSessionId ? (messages[activeSessionId] || []) : []

  return (
    <div style={{ minHeight: '100%', background: C.bg, overflowX: 'hidden' }}>
      <TopBar tab={tab} owned={owned} setTab={setTab}/>

      <div>
        {tab === 'browse'   && <BrowseScreen owned={owned} onSelect={openDetail}/>}
        {tab === 'myagents' && <MyAgentsScreen owned={owned} agents={AGENTS} onChat={openChat} onUnsubscribe={unsubscribe} onBrowse={() => setTab('browse')}/>}
      </div>

      {!isDesktop && <BottomNav tab={tab} setTab={setTab} owned={owned}/>}

      {/* Detail modal */}
      {detailAgent && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', background: 'rgba(28,25,22,0.5)', backdropFilter: 'blur(6px)', padding: '0 0 32px' }}
          onClick={closeDetail}>
          <div onClick={e => e.stopPropagation()}
            style={{ background: C.bg, borderRadius: 20, width: '100%', maxWidth: 760, maxHeight: '78vh', display: 'flex', flexDirection: 'column', animation: 'sheetup 0.28s ease', overflow: 'hidden' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px', borderBottom: `1px solid ${C.line}`, flexShrink: 0 }}>
              <span style={{ fontFamily: "'DM Serif Display',serif", fontSize: 18, color: C.ink }}>{detailAgent.name}</span>
              <button onClick={closeDetail} style={{ width: 32, height: 32, borderRadius: 16, border: 'none', background: C.subtle, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted, fontSize: 18, lineHeight: 1 }}>×</button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', position: 'relative' }}>
              <DetailScreen agent={detailAgent} owned={owned.includes(detailAgent.id)} onSubscribe={() => setSubscribeModal(true)} onUnsubscribe={() => unsubscribe(detailAgent.id)} onChat={() => openChat(detailAgent)}/>
            </div>
          </div>
        </div>
      )}

      {/* Chat — fullscreen */}
      {chatAgent && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', flexDirection: 'column', background: CC.bg, animation: 'fadein 0.18s ease' }}>
          {/* Top bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '0 20px', height: 56, borderBottom: `1px solid ${CC.line}`, flexShrink: 0, background: CC.card }}>
            <button onClick={closeChat} style={{ background: 'none', border: 'none', color: CC.accent, cursor: 'pointer', fontSize: 22, padding: '2px 8px 2px 0', lineHeight: 1, fontFamily: 'inherit', flexShrink: 0 }}>‹</button>
            <span style={{ fontFamily: "'DM Serif Display',serif", fontSize: 21, letterSpacing: '-0.3px', color: CC.ink, flex: 1 }}>Chat with {chatAgent.name}</span>
            <button onClick={() => setChatDark(d => !d)}
              style={{ width: 32, height: 32, borderRadius: 8, border: `1.5px solid ${CC.line}`, background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: CC.muted, flexShrink: 0, transition: 'background 0.15s' }}
              onMouseEnter={e => e.currentTarget.style.background = CC.subtle}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              title={chatDark ? 'Light mode' : 'Dark mode'}>
              {chatDark
                ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="4"/><line x1="12" y1="20" x2="12" y2="22"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="2" y1="12" x2="4" y2="12"/><line x1="20" y1="12" x2="22" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
                : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>
              }
            </button>
          </div>
          {/* Body: sidebar + chat */}
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            {isDesktop && (sidebarOpen ? (
              <div style={{ width: sidebarWidth, flexShrink: 0, background: CC.sidebar || CC.card, borderRight: `1px solid ${CC.line}`, display: 'flex', flexDirection: 'column', position: 'relative' }}>
                <div style={{ padding: '16px 14px 12px', display: 'flex', gap: 8 }}>
                  <button onClick={newChat}
                    style={{ flex: 1, padding: '10px 14px', borderRadius: 10, border: `1.5px solid ${CC.line}`, background: 'transparent', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600, fontSize: 13, color: CC.ink, display: 'flex', alignItems: 'center', gap: 8, transition: 'background 0.15s' }}
                    onMouseEnter={e => e.currentTarget.style.background = CC.subtle}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <span style={{ fontSize: 18, lineHeight: 1 }}>+</span> New chat
                  </button>
                  <button onClick={() => setSidebarOpen(false)}
                    style={{ width: 38, height: 38, borderRadius: 10, border: `1.5px solid ${CC.line}`, background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: CC.muted, flexShrink: 0, transition: 'background 0.15s' }}
                    onMouseEnter={e => e.currentTarget.style.background = CC.subtle}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                    title="Collapse panel">
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                      <line x1="11" y1="2" x2="11" y2="13"/><polyline points="7,5 4,7.5 7,10"/>
                    </svg>
                  </button>
                </div>
                <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px 16px' }}>
                  {agentSessions.length === 0 && (
                    <div style={{ padding: '12px 8px', color: CC.faint, fontSize: 12 }}>No conversations yet</div>
                  )}
                  {agentSessions.map(s => (
                    <SessionRow key={s.id} s={s} active={s.id === activeSessionId} C={CC}
                      onSelect={() => selectSession(s.id)}
                      onDelete={() => deleteSession(s.id)}
                      onRename={renameSession}/>
                  ))}
                </div>
                {owned.includes(chatAgent.id) && (
                  <div style={{ padding: '12px 14px 16px' }}>
                    <button onClick={unsubscribeFromChat}
                      style={{ width: '100%', padding: '10px 12px', borderRadius: 10, border: `1.5px solid ${CC.line}`, background: 'transparent', color: CC.faint, cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600, fontSize: 13, transition: 'background 0.15s,color 0.15s' }}
                      onMouseEnter={e => { e.currentTarget.style.background = CC.subtle; e.currentTarget.style.color = CC.accent }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = CC.faint }}>
                      Unsubscribe from {chatAgent.name}
                    </button>
                  </div>
                )}
                <div onMouseDown={onDragStart} ref={dragRef}
                  style={{ position: 'absolute', top: 0, right: -3, width: 6, height: '100%', cursor: 'col-resize', zIndex: 10 }}/>
              </div>
            ) : (
              <div style={{ width: 48, flexShrink: 0, background: CC.sidebar || CC.card, borderRight: `1px solid ${CC.line}`, display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '14px 0' }}>
                <button onClick={() => setSidebarOpen(true)}
                  style={{ width: 32, height: 32, borderRadius: 8, border: `1.5px solid ${CC.line}`, background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: CC.muted, transition: 'background 0.15s' }}
                  onMouseEnter={e => e.currentTarget.style.background = CC.subtle}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  title="Expand panel">
                  <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                    <line x1="4" y1="2" x2="4" y2="13"/><polyline points="8,5 11,7.5 8,10"/>
                  </svg>
                </button>
              </div>
            ))}
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <ChatScreen
                agent={chatAgent}
                history={activeHistory}
                sessionId={activeSessionId}
                onAddMsg={addMsg}
                onNeedSession={() => { const id = createSession(chatAgent); setActiveSessionId(id); return id }}
                colors={CC}/>
            </div>
          </div>
        </div>
      )}

      {subscribeModal && detailAgent && (
        <SubscribeModal agent={detailAgent} onClose={() => setSubscribeModal(false)} onConfirm={subscribe} desktop={isDesktop}/>
      )}
    </div>
  )
}
