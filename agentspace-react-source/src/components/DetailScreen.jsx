import { useState } from 'react'
import { REVIEWS } from '../data/agents'
import AgentIcon from './AgentIcon'
import Stars from './Stars'
import { useC } from '../context/ThemeContext'

function RatingBar({ pct, color }) {
  const C = useC()
  return (
    <div style={{ flex: 1, height: 6, background: C.subtle, borderRadius: 3, overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }}/>
    </div>
  )
}

export default function DetailScreen({ agent, owned, onSubscribe, onUnsubscribe, onChat }) {
  const C = useC()
  const comingSoon = agent.comingSoon
  const agentReviews = REVIEWS[agent.id] || []
  const [reviewsExpanded, setReviewsExpanded] = useState(false)
  const visible = reviewsExpanded ? agentReviews : agentReviews.slice(0, 2)
  const dist = [{ stars: 5, pct: agent.rating >= 4.8 ? 82 : 68 }, { stars: 4, pct: agent.rating >= 4.8 ? 12 : 20 }, { stars: 3, pct: 4 }, { stars: 2, pct: 1 }, { stars: 1, pct: 1 }]

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      <div className="sy" style={{ flex: 1, paddingBottom: 90 }}>
        {/* Hero */}
        <div style={{ background: C.hero, padding: '28px 24px 36px', color: '#fff', position: 'relative' }}>
          <div style={{ position: 'absolute', right: -24, bottom: -24, width: 130, height: 130, borderRadius: 65, background: `${agent.c1}22`, border: `1px solid ${agent.c1}33`, pointerEvents: 'none' }}/>
          <AgentIcon agent={agent} size={76} pfx="det"/>
          <div style={{ marginTop: 16, fontFamily: "'DM Serif Display',serif", fontSize: 36, letterSpacing: '-0.5px', lineHeight: 1.05 }}>{agent.name}</div>
          <div style={{ color: agent.c2, fontWeight: 600, fontSize: 15, marginTop: 5 }}>{agent.role}</div>
          <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: 13, marginTop: 2 }}>{agent.spec}</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
            <Stars n={agent.rating}/>
            <span style={{ color: 'rgba(255,255,255,0.55)', fontSize: 13 }}>{agent.rating} · {agent.reviews.toLocaleString()} reviews · {agent.sessions} sessions</span>
          </div>
          <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap', marginTop: 14 }}>
            {comingSoon
              ? <span style={{ background: 'rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.75)', padding: '4px 11px', borderRadius: 20, fontSize: 12, fontWeight: 500 }}>Coming soon</span>
              : agent.tags.map(t => <span key={t} style={{ background: 'rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.75)', padding: '4px 11px', borderRadius: 20, fontSize: 12, fontWeight: 500 }}>{t}</span>)
            }
          </div>
        </div>

        <div style={{ padding: '22px 24px' }}>
          {/* About */}
          <div style={{ background: C.card, borderRadius: 16, padding: '20px', marginBottom: 14, border: `1px solid ${C.line}` }}>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, color: C.faint, textTransform: 'uppercase', marginBottom: 10 }}>About {agent.name}</div>
            <div style={{ color: C.mid, fontSize: 15, lineHeight: 1.7 }}>
              {comingSoon ? `${agent.name} is being prepared for AgentSpace and is not available to subscribe yet.` : agent.bio}
            </div>
          </div>

          {/* Can help with */}
          <div style={{ background: C.card, borderRadius: 16, padding: '20px', marginBottom: 14, border: `1px solid ${C.line}` }}>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, color: C.faint, textTransform: 'uppercase', marginBottom: 12 }}>What {agent.name} can help with</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {(comingSoon ? [`${agent.name}'s profile and chat experience are coming soon.`] : agent.canHelp).map((item, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 11 }}>
                  <div style={{ width: 20, height: 20, borderRadius: 10, background: `${agent.c1}18`, border: `1.5px solid ${agent.c1}40`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: 1 }}>
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><polyline points="1.5,5.5 4,7.5 8.5,2.5" stroke={agent.c1} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  </div>
                  <span style={{ color: C.mid, fontSize: 14, lineHeight: 1.5 }}>{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* What to expect */}
          <div style={{ background: `${agent.c1}0d`, borderRadius: 16, padding: '18px 20px', marginBottom: 14, border: `1.5px solid ${agent.c1}22` }}>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, color: agent.c1, textTransform: 'uppercase', marginBottom: 8 }}>What to expect</div>
            <div style={{ color: C.mid, fontSize: 14, lineHeight: 1.65 }}>
              {comingSoon ? 'You can preview this agent now. Subscriptions and chat will unlock when the agent launches.' : agent.expect}
            </div>
          </div>

          {/* Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginBottom: 14 }}>
            {[['24/7','Available'],[agent.reviews.toLocaleString(),'Reviews'],['< 2 min','Response']].map(([v,l]) => (
              <div key={l} style={{ background: C.card, borderRadius: 14, padding: '16px 12px', textAlign: 'center', border: `1px solid ${C.line}` }}>
                <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 22, color: C.ink, lineHeight: 1 }}>{v}</div>
                <div style={{ color: C.faint, fontSize: 12, marginTop: 4 }}>{l}</div>
              </div>
            ))}
          </div>

          {/* Pricing */}
          <div style={{ background: C.card, borderRadius: 16, padding: '20px', border: `2px solid ${agent.c1}28`, marginBottom: 14 }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
              <div>
                <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 42, color: C.ink, lineHeight: 1 }}>${agent.price}</div>
                <div style={{ color: C.muted, fontSize: 14, marginTop: 4 }}>per month</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ color: C.muted, fontSize: 13 }}>{comingSoon ? 'Not yet available' : 'Unlimited messages'}</div>
                <div style={{ color: C.muted, fontSize: 13 }}>{comingSoon ? 'Launch details soon' : 'Cancel anytime'}</div>
              </div>
            </div>
          </div>

          {/* Reviews */}
          <div style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 22, color: C.ink }}>Reviews</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Stars n={agent.rating}/>
                <span style={{ fontWeight: 700, fontSize: 15, color: C.ink }}>{agent.rating}</span>
                <span style={{ color: C.faint, fontSize: 13 }}>({agent.reviews.toLocaleString()})</span>
              </div>
            </div>
            <div style={{ background: C.card, borderRadius: 16, padding: '18px', marginBottom: 14, border: `1px solid ${C.line}` }}>
              {dist.map(d => (
                <div key={d.stars} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ color: C.muted, fontSize: 12, width: 8, flexShrink: 0 }}>{d.stars}</span>
                  <span style={{ color: '#d97706', fontSize: 12, flexShrink: 0 }}>★</span>
                  <RatingBar pct={d.pct} color={agent.c1}/>
                  <span style={{ color: C.faint, fontSize: 12, width: 28, textAlign: 'right', flexShrink: 0 }}>{d.pct}%</span>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {visible.map((r, i) => (
                <div key={i} style={{ background: C.card, borderRadius: 16, padding: '18px', border: `1px solid ${C.line}`, animation: 'fadein 0.3s ease' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ width: 34, height: 34, borderRadius: 17, background: `${agent.c1}18`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 13, color: agent.c1, flexShrink: 0 }}>{r.name.charAt(0)}</div>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: 14, color: C.ink }}>{r.name}</div>
                        <div style={{ color: C.faint, fontSize: 12 }}>{r.date}</div>
                      </div>
                    </div>
                    <Stars n={r.rating}/>
                  </div>
                  <div style={{ color: C.mid, fontSize: 14, lineHeight: 1.6, fontStyle: 'italic' }}>"{r.text}"</div>
                </div>
              ))}
            </div>
            {agentReviews.length > 2 && (
              <button onClick={() => setReviewsExpanded(e => !e)}
                style={{ marginTop: 10, width: '100%', padding: '12px', borderRadius: 12, border: `1.5px solid ${C.line}`, cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600, fontSize: 14, background: 'transparent', color: C.muted, transition: 'all 0.15s' }}
                onMouseEnter={e => { e.currentTarget.style.background = C.subtle; e.currentTarget.style.color = C.ink }}
                onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = C.muted }}>
                {reviewsExpanded ? 'Show fewer reviews' : `See all ${agentReviews.length} reviews`}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Sticky CTA */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '14px 24px 20px', background: `linear-gradient(to top,${C.bg} 65%,transparent)` }}>
        {comingSoon ? (
          <button disabled style={{ width: '100%', padding: '16px', borderRadius: 14, border: 'none', fontFamily: "'DM Sans',sans-serif", fontWeight: 700, fontSize: 17, background: C.subtle, color: C.faint, cursor: 'not-allowed' }}>Coming soon</button>
        ) : owned ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <button onClick={onChat} style={{ width: '100%', padding: '16px', borderRadius: 14, border: 'none', cursor: 'pointer', fontFamily: "'DM Sans',sans-serif", fontWeight: 700, fontSize: 17, background: C.ink, color: C.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, transition: 'opacity 0.15s' }} onMouseEnter={e => e.currentTarget.style.opacity = '0.85'} onMouseLeave={e => e.currentTarget.style.opacity = '1'}>Chat with {agent.name} <span style={{ fontSize: 20 }}>→</span></button>
            <button onClick={onUnsubscribe} style={{ width: '100%', padding: '10px', borderRadius: 14, border: 'none', cursor: 'pointer', fontFamily: "'DM Sans',sans-serif", fontWeight: 500, fontSize: 14, background: 'transparent', color: C.faint, transition: 'color 0.15s' }} onMouseEnter={e => e.currentTarget.style.color = C.accent} onMouseLeave={e => e.currentTarget.style.color = C.faint}>Unsubscribe</button>
          </div>
        ) : (
          <button onClick={onSubscribe} style={{ width: '100%', padding: '16px', borderRadius: 14, border: 'none', cursor: 'pointer', fontFamily: "'DM Sans',sans-serif", fontWeight: 700, fontSize: 17, background: C.accent, color: '#fff', transition: 'opacity 0.15s' }} onMouseEnter={e => e.currentTarget.style.opacity = '0.85'} onMouseLeave={e => e.currentTarget.style.opacity = '1'}>Subscribe · ${agent.price}/mo</button>
        )}
      </div>
    </div>
  )
}
