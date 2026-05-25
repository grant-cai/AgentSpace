import { useState, useEffect, useRef } from 'react'
import { REPLIES } from '../data/agents'
import AgentIcon from './AgentIcon'
import { useC } from '../context/ThemeContext'

function greeting() {
  const h = new Date().getHours()
  if (h >= 5  && h < 12) return 'Good morning'
  if (h >= 12 && h < 17) return 'Good afternoon'
  if (h >= 17 && h < 21) return 'Good evening'
  return 'Good night'
}

export default function ChatScreen({ agent, history, sessionId, onAddMsg, onNeedSession, colors }) {
  const globalC = useC()
  const C = colors || globalC
  const [input, setInput] = useState('')
  const [typing, setTyping] = useState(false)
  const [showGreeting, setShowGreeting] = useState(true)
  const scrollRef = useRef(null)
  const textareaRef = useRef(null)
  const prevSessionIdRef = useRef(sessionId)

  useEffect(() => {
    const prev = prevSessionIdRef.current
    prevSessionIdRef.current = sessionId
    if (prev !== null && prev !== sessionId) setShowGreeting(true)
    textareaRef.current?.focus()
  }, [sessionId])

  useEffect(() => { textareaRef.current?.focus() }, [])

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 140) + 'px'
  }, [input])

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight + 9999
  }, [history, typing])

  const send = async () => {
  if (!input.trim()) return
  const txt = input.trim()
  setInput('')
  setShowGreeting(false)
  if (textareaRef.current) textareaRef.current.style.height = 'auto'
  const sid = sessionId || onNeedSession?.()
  onAddMsg(sid, { role: 'user', text: txt })
  setTyping(true)

  try {
    const res = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: txt, thread_id: sid }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || 'Error')
    onAddMsg(sid, { role: 'agent', text: data.response })
  } catch (err) {
    onAddMsg(sid, { role: 'agent', text: `⚠️ ${err.message}` })
  } finally {
    setTyping(false)
  }
}

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: C.bg }}>
      {/* Chat header */}
      <div style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', gap: 14, background: C.card, borderBottom: `1px solid ${C.line}`, flexShrink: 0 }}>
        <AgentIcon agent={agent} size={40} pfx="ch"/>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 16, color: C.ink }}>{agent.name}</div>
          <div style={{ marginTop: 2 }}>
            <span style={{ color: C.faint, fontSize: 12 }}>{agent.role}</span>
          </div>
        </div>
      </div>

      {showGreeting && history.length === 0 ? (
        /* Empty state — time-based greeting */
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 24px', gap: 10 }}>
          <AgentIcon agent={agent} size={64} pfx="greet"/>
          <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 26, color: C.ink, marginTop: 8 }}>{greeting()}</div>
          <div style={{ color: C.muted, fontSize: 14, textAlign: 'center', maxWidth: 280 }}>
            How can {agent.name} help you today?
          </div>
        </div>
      ) : (
        /* Messages */
        <div ref={scrollRef} className="sy" style={{ flex: 1, padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ textAlign: 'center', marginBottom: 4 }}>
            <span style={{ background: C.subtle, color: C.mid, fontSize: 11, fontWeight: 600, padding: '4px 12px', borderRadius: 20, letterSpacing: 0.4 }}>Today</span>
          </div>
          {history.map((m, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', animation: 'fadein 0.3s ease' }}>
              <div style={{ maxWidth: '74%', padding: '11px 15px', fontSize: 14, lineHeight: 1.6, borderRadius: m.role === 'user' ? '18px 18px 4px 18px' : '4px 18px 18px 18px', background: m.role === 'user' ? C.ink : C.card, color: m.role === 'user' ? C.bg : C.mid, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', border: m.role === 'user' ? 'none' : `1px solid ${C.line}`, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{m.text}</div>
            </div>
          ))}
          {typing && (
            <div style={{ display: 'flex' }}>
              <div style={{ padding: '12px 16px', borderRadius: '4px 18px 18px 18px', background: C.card, border: `1px solid ${C.line}`, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', display: 'flex', gap: 5, alignItems: 'center' }}>
                {[0,1,2].map(i => <div key={i} style={{ width: 7, height: 7, borderRadius: 4, background: C.faint, animation: `bounce 1.1s ${i * 0.18}s ease-in-out infinite` }}/>)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Input bar */}
      <div style={{ padding: '12px 20px 18px', background: C.card, borderTop: `1px solid ${C.line}`, flexShrink: 0, display: 'flex', gap: 10 }}>
        <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          placeholder={`Message ${agent.name}…`} rows={1}
          style={{ flex: 1, background: C.bg, border: `1.5px solid ${C.line}`, borderRadius: 18, padding: '12px 18px', color: C.ink, fontSize: 14, fontFamily: 'inherit', outline: 'none', transition: 'border-color 0.15s', resize: 'none', lineHeight: 1.5, overflowY: 'auto' }}
          onFocus={e => e.target.style.borderColor = C.ink}
          onBlur={e => e.target.style.borderColor = C.line}/>
        <button onClick={send} style={{ width: 46, height: 46, borderRadius: 23, border: 'none', cursor: 'pointer', background: C.ink, color: C.bg, fontSize: 20, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'opacity 0.15s' }} onMouseEnter={e => e.currentTarget.style.opacity = '0.8'} onMouseLeave={e => e.currentTarget.style.opacity = '1'}>→</button>
      </div>
    </div>
  )
}
