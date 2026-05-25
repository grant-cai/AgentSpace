import AgentIcon from './AgentIcon'
import Stars from './Stars'
import { useC } from '../context/ThemeContext'

export default function MyAgentsScreen({ owned, agents, onChat, onUnsubscribe, onBrowse }) {
  const C = useC()
  if (owned.length === 0) return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: 40, textAlign: 'center' }}>
      <div style={{ width: 80, height: 80, borderRadius: 40, background: C.subtle, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke={C.faint} strokeWidth="2" strokeLinecap="round"><circle cx="16" cy="16" r="12"/><line x1="16" y1="10" x2="16" y2="22"/><line x1="10" y1="16" x2="22" y2="16"/></svg>
      </div>
      <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 24, marginBottom: 8, color: C.ink }}>No agents yet</div>
      <div style={{ color: C.muted, fontSize: 14, lineHeight: 1.6, marginBottom: 28, maxWidth: 280 }}>Browse the marketplace to find an AI expert and subscribe for unlimited access.</div>
      <button onClick={onBrowse} style={{ padding: '14px 28px', borderRadius: 12, border: 'none', cursor: 'pointer', fontFamily: "'DM Sans',sans-serif", fontWeight: 700, fontSize: 16, background: C.ink, color: C.bg }}>Browse Agents</button>
    </div>
  )

  return (
    <div style={{ padding: '24px 24px' }}>
      <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 28, letterSpacing: '-0.4px', marginBottom: 4 }}>My Agents</div>
      <div style={{ color: C.muted, fontSize: 14, marginBottom: 22 }}>{owned.length} active subscription{owned.length !== 1 ? 's' : ''}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {agents.filter(a => owned.includes(a.id)).map(a => (
          <div key={a.id} style={{ background: C.card, borderRadius: 18, padding: '18px', border: `1px solid ${C.line}`, display: 'flex', gap: 14, alignItems: 'center', animation: 'fadein 0.35s ease' }}>
            <AgentIcon agent={a} size={54} pfx="my"/>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 600, fontSize: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                {a.name}
                <span style={{ display: 'inline-flex', alignItems: 'center', padding: '3px 9px', borderRadius: 20, fontSize: 11, fontWeight: 700, letterSpacing: 0.4, background: '#dcfce7', color: C.green }}>Active</span>
              </div>
              <div style={{ color: C.muted, fontSize: 13, marginTop: 2 }}>{a.role} · {a.spec}</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 5 }}>
                <Stars n={a.rating}/><span style={{ color: C.faint, fontSize: 12 }}>{a.rating}</span>
              </div>
            </div>
            <div style={{ flexShrink: 0, textAlign: 'right', display: 'flex', flexDirection: 'column', gap: 6 }}>
              <button onClick={() => onChat(a)} style={{ padding: '10px 18px', borderRadius: 10, border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 700, fontSize: 14, background: C.ink, color: C.bg, transition: 'opacity 0.15s' }} onMouseEnter={e => e.currentTarget.style.opacity = '0.8'} onMouseLeave={e => e.currentTarget.style.opacity = '1'}>Chat →</button>
              <button onClick={() => onUnsubscribe(a.id)} style={{ padding: '4px 0', border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 500, fontSize: 12, background: 'transparent', color: C.faint, transition: 'color 0.15s' }} onMouseEnter={e => e.currentTarget.style.color = C.accent} onMouseLeave={e => e.currentTarget.style.color = C.faint}>Unsubscribe</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
