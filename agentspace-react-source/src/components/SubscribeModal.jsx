import AgentIcon from './AgentIcon'
import { useC } from '../context/ThemeContext'

export default function SubscribeModal({ agent, onClose, onConfirm, desktop }) {
  const C = useC()
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(28,25,22,0.4)', backdropFilter: 'blur(8px)', zIndex: 200, display: 'flex', alignItems: desktop ? 'center' : 'flex-end', justifyContent: 'center' }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ background: C.bg, borderRadius: desktop ? '20px' : '22px 22px 0 0', padding: '26px 26px 40px', width: desktop ? '440px' : '100%', maxWidth: '100%', boxShadow: '0 -4px 40px rgba(0,0,0,0.12)', animation: desktop ? 'fadein 0.2s ease' : 'sheetup 0.28s ease' }}>
        {!desktop && <div style={{ width: 36, height: 4, borderRadius: 2, background: C.subtle, margin: '0 auto 24px' }}/>}
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 24 }}>
          <AgentIcon agent={agent} size={58} pfx="mod"/>
          <div>
            <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 26, letterSpacing: '-0.4px' }}>{agent.name}</div>
            <div style={{ color: C.muted, fontSize: 14, marginTop: 2 }}>{agent.role} · {agent.spec}</div>
          </div>
        </div>
        <div style={{ background: C.card, border: `2px solid ${agent.c1}25`, borderRadius: 16, padding: '20px 22px', marginBottom: 20 }}>
          <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 38, letterSpacing: '-0.5px', color: C.ink, lineHeight: 1 }}>
            ${agent.price}<span style={{ fontFamily: "'DM Sans',sans-serif", fontWeight: 400, fontSize: 16, color: C.muted }}>/month</span>
          </div>
          <div style={{ height: '1px', background: C.subtle, margin: '16px 0' }}/>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {['Unlimited conversations, 24/7','Personalized to your goals','Priority response time','Cancel or pause anytime'].map(f => (
              <div key={f} style={{ display: 'flex', alignItems: 'center', gap: 11, color: C.mid, fontSize: 14 }}>
                <span style={{ color: agent.c1, fontWeight: 700, fontSize: 16, lineHeight: 1 }}>✓</span>{f}
              </div>
            ))}
          </div>
        </div>
        <button onClick={onConfirm} style={{ width: '100%', padding: '17px', borderRadius: 14, border: 'none', cursor: 'pointer', fontFamily: "'DM Sans',sans-serif", fontWeight: 700, fontSize: 18, background: C.accent, color: '#fff', transition: 'opacity 0.15s' }} onMouseEnter={e => e.currentTarget.style.opacity = '0.88'} onMouseLeave={e => e.currentTarget.style.opacity = '1'}>Subscribe Now</button>
        <div style={{ textAlign: 'center', color: C.faint, fontSize: 12, marginTop: 12 }}>Secure checkout · Cancel anytime · No hidden fees</div>
      </div>
    </div>
  )
}
