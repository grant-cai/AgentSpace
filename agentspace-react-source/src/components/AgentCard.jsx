import { useState } from 'react'
import AgentIcon from './AgentIcon'
import Stars from './Stars'
import { useC } from '../context/ThemeContext'

export default function AgentCard({ agent, owned, onClick }) {
  const C = useC()
  const [hov, setHov] = useState(false)
  const soon = agent.comingSoon

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: C.card, borderRadius: 20, cursor: 'pointer', overflow: 'hidden',
        boxShadow: hov ? '0 8px 28px rgba(0,0,0,0.11),0 2px 6px rgba(0,0,0,0.05)' : '0 1px 4px rgba(0,0,0,0.07),0 2px 8px rgba(0,0,0,0.04)',
        border: '1px solid rgba(0,0,0,0.06)', transition: 'box-shadow 0.2s,transform 0.18s',
        transform: hov ? 'translateY(-2px)' : 'none', animation: 'fadein 0.3s ease', display: 'flex', flexDirection: 'column',
        opacity: soon ? 0.6 : 1,
      }}
    >
      <div style={{ background: `linear-gradient(135deg,${agent.c1},${agent.c2})`, padding: '22px 18px 18px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', right: -16, bottom: -16, width: 70, height: 70, borderRadius: 35, background: 'rgba(255,255,255,0.1)', pointerEvents: 'none' }}/>
        <AgentIcon agent={agent} size={48} pfx="gc"/>
        <div style={{ position: 'absolute', top: 12, right: 12 }}>
          {owned && <span style={{ display: 'inline-flex', alignItems: 'center', padding: '3px 9px', borderRadius: 20, fontSize: 11, fontWeight: 700, letterSpacing: 0.4, background: 'rgba(255,255,255,0.22)', color: '#fff' }}>Active</span>}
          {soon && <span style={{ display: 'inline-flex', alignItems: 'center', padding: '3px 9px', borderRadius: 20, fontSize: 11, fontWeight: 700, letterSpacing: 0.4, background: 'rgba(0,0,0,0.25)', color: 'rgba(255,255,255,0.8)' }}>Soon</span>}
        </div>
      </div>
      <div style={{ padding: '14px 16px 16px', flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ fontWeight: 700, fontSize: 16, color: C.ink, lineHeight: 1.2 }}>{agent.name}</div>
        <div style={{ color: C.muted, fontSize: 12 }}>{agent.role}</div>
        <div style={{ color: C.faint, fontSize: 11, marginTop: 1 }}>{agent.spec}</div>
        {!soon ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 6 }}>
              <Stars n={agent.rating}/>
              <span style={{ color: C.faint, fontSize: 11 }}>{agent.rating}</span>
              <span style={{ color: C.subtle, fontSize: 11, marginLeft: 2 }}>({agent.reviews.toLocaleString()})</span>
            </div>
            <div style={{ marginTop: 'auto', paddingTop: 12, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              {owned
                ? <span style={{ fontSize: 13, fontWeight: 700, color: C.green }}>Subscribed</span>
                : <span style={{ fontSize: 13, fontWeight: 700, color: C.accent }}>Subscribe now</span>
              }
              <span style={{ color: C.muted, fontSize: 11 }}>{agent.sessions} sessions</span>
            </div>
          </>
        ) : (
          <div style={{ marginTop: 'auto', paddingTop: 12 }}>
            <span style={{ fontSize: 13, color: C.faint }}>Coming soon</span>
          </div>
        )}
      </div>
    </div>
  )
}
