import { useState } from 'react'
import { ALL_AGENTS, CATS } from '../data/agents'
const AGENTS = ALL_AGENTS
import AgentIcon from './AgentIcon'
import AgentCard from './AgentCard'
import { useC } from '../context/ThemeContext'

export default function BrowseScreen({ owned, onSelect }) {
  const C = useC()
  const [cat, setCat] = useState('All')
  const [q, setQ] = useState('')
  const list = AGENTS.filter(a => (cat === 'All' || a.cat === cat) && (a.name + a.role + a.spec).toLowerCase().includes(q.toLowerCase()))

  return (
    <div>
      {/* Hero */}
      <div style={{ background: C.hero, padding: '36px 24px 28px', color: '#fff', position: 'relative' }}>
        <div style={{ position: 'absolute', top: -30, right: -30, width: 150, height: 150, borderRadius: 75, border: '1px solid rgba(255,255,255,0.06)', pointerEvents: 'none' }}/>
        <div style={{ position: 'absolute', top: 20, right: 20, width: 80, height: 80, borderRadius: 40, border: '1px solid rgba(255,255,255,0.04)', pointerEvents: 'none' }}/>
        <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 'clamp(28px,5vw,40px)', lineHeight: 1.1, letterSpacing: '-0.5px', position: 'relative' }}>Expert advice,<br/><em>on demand.</em></div>
        <div style={{ position: 'relative', marginTop: 20 }}>
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search by name, role, or specialty…"
            style={{ width: '100%', background: 'rgba(255,255,255,0.09)', border: '1px solid rgba(255,255,255,0.14)', borderRadius: 12, padding: '12px 16px 12px 42px', color: '#fff', fontSize: 14, fontFamily: 'inherit', outline: 'none', transition: 'border-color 0.15s' }}
            onFocus={e => e.target.style.borderColor = 'rgba(255,255,255,0.3)'}
            onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.14)'}/>
          <span style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.28)', fontSize: 18, pointerEvents: 'none' }}>⌕</span>
        </div>
      </div>

      {/* Categories */}
      <div className="sx" style={{ display: 'flex', gap: 7, padding: '16px 24px', borderBottom: `1px solid ${C.line}` }}>
        {CATS.map(c => (
          <button key={c} onClick={() => setCat(c)}
            style={{ flexShrink: 0, padding: '7px 15px', borderRadius: 8, border: cat === c ? `2px solid ${C.ink}` : '1.5px solid #d4cdc4', cursor: 'pointer', fontSize: 13, fontWeight: cat === c ? 700 : 400, fontFamily: 'inherit', background: cat === c ? C.ink : 'transparent', color: cat === c ? C.bg : C.muted, transition: 'all 0.14s', whiteSpace: 'nowrap' }}>{c}</button>
        ))}
      </div>

      <div style={{ padding: '20px 24px 32px' }}>
        {/* Featured banner */}
        {!q && cat === 'All' && (
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, color: C.faint, textTransform: 'uppercase', marginBottom: 12 }}>Featured this week</div>
            <div onClick={() => onSelect(AGENTS[0])} style={{ background: C.ink, borderRadius: 20, padding: '20px 22px', cursor: 'pointer', position: 'relative', overflow: 'hidden', display: 'flex', gap: 16, alignItems: 'center' }}>
              <div style={{ position: 'absolute', right: -20, top: -20, width: 120, height: 120, borderRadius: 60, background: `${AGENTS[0].c1}25`, pointerEvents: 'none' }}/>
              <AgentIcon agent={AGENTS[0]} size={60} pfx="feat"/>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 10, color: AGENTS[0].c2, fontWeight: 700, letterSpacing: 1.2, marginBottom: 4 }}>TOP RATED · {AGENTS[0].sessions} SESSIONS</div>
                <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 24, color: '#fff', lineHeight: 1.1 }}>{AGENTS[0].name}</div>
                <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13, marginTop: 3 }}>{AGENTS[0].role} · {AGENTS[0].spec}</div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 26, color: '#fff' }}>${AGENTS[0].price}</div>
                <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: 12 }}>/month</div>
              </div>
            </div>
          </div>
        )}

        {/* Grid */}
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, color: C.faint, textTransform: 'uppercase', marginBottom: 14 }}>
          {cat === 'All' ? 'All Agents' : cat} · {list.length}
        </div>
        {list.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 24px', color: C.muted }}>
            <div style={{ width: 56, height: 56, borderRadius: 28, background: C.subtle, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={C.faint} strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="7"/><line x1="17" y1="17" x2="21" y2="21"/></svg>
            </div>
            <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 20, marginBottom: 6, color: C.ink }}>No results</div>
            <div style={{ fontSize: 14 }}>Try a different search or category</div>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(220px,1fr))', gap: 14 }}>
            {list.map(a => (
              <AgentCard key={a.id} agent={a} owned={owned.includes(a.id)} onClick={() => onSelect(a)}/>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
