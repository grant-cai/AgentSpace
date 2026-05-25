import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { AGENTS } from '../data/agents'
import AgentIcon from '../components/AgentIcon'
import '../styles/home.css'

const grant = { name: 'Grant', c1: '#0891b2', c2: '#22d3ee' }


export default function Home() {
  const heroRef = useRef(null)
  const navRef = useRef(null)
  const [formState, setFormState] = useState({ first: '', last: '', email: '', profession: '', spec: '', bio: '' })
  const [formError, setFormError] = useState({})
  const [submitted, setSubmitted] = useState(false)

  // Hero load animation
  useEffect(() => {
    const t = setTimeout(() => heroRef.current?.classList.add('hero-loaded'), 80)
    return () => clearTimeout(t)
  }, [])

  // Nav scroll
  useEffect(() => {
    const onScroll = () => navRef.current?.classList.toggle('scrolled', window.scrollY > 40)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Scroll reveal
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('revealed'); observer.unobserve(e.target) } })
    }, { threshold: 0.12 })
    document.querySelectorAll('.reveal').forEach(el => observer.observe(el))
    return () => observer.disconnect()
  }, [])


  const handleSubmit = () => {
    const errors = {}
    if (!formState.first.trim()) errors.first = true
    if (!formState.email.trim()) errors.email = true
    if (!formState.profession) errors.profession = true
    if (Object.keys(errors).length > 0) { setFormError(errors); setTimeout(() => setFormError({}), 2000); return }
    setSubmitted(true)
  }

  return (
    <>
      <nav ref={navRef} id="nav">
        <Link to="/" className="nav-logo">AgentSpace</Link>
        <div className="nav-links">
          <a href="#how-it-works" className="nav-link">How it works</a>
          <a href="#for-professionals" className="nav-link">For professionals</a>
          <Link to="/browse" className="nav-link nav-cta">Browse Agents →</Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero" id="hero" ref={heroRef}>
        <div className="hero-orb hero-orb-1"/>
        <div className="hero-orb hero-orb-2"/>
        <div className="hero-ring" style={{ width: 900, height: 900, right: -300, top: -300 }}/>
        <div className="hero-ring" style={{ width: 500, height: 500, right: -100, top: -100 }}/>
        <div className="hero-eyebrow">Introducing AgentSpace</div>
        <h1 className="hero-headline">The world's experts,<br/><em>available now.</em></h1>
        <p className="hero-sub">Meet Grant — an AI writing tutor built from real teaching expertise. Get unlimited help with essays, research papers, personal statements, and more. Expert feedback whenever you need it, at a fraction of the cost.</p>
        <div className="hero-actions">
          <Link to="/browse" className="btn-primary">Enter AgentSpace <span style={{ fontSize: 18, lineHeight: 1 }}>→</span></Link>
          <a href="#for-professionals" className="btn-ghost">I'm a professional</a>
        </div>
        <div className="hero-agents-strip" style={{ gap: 8 }}>
          <div className="agent-thumbs">
            <AgentIcon agent={AGENTS[0]} size={32} pfx="herostrip"/>
          </div>
          <span className="hero-strip-text"><strong>Grant</strong> — your AI writing tutor, available 24/7</span>
        </div>
      </section>

      {/* How it works */}
      <section className="hiw" id="how-it-works">
        <div className="container">
          <div className="reveal">
            <div className="section-eyebrow">How it works</div>
            <h2 className="section-title">Expert advice in<br/>three simple steps</h2>
            <p className="section-sub">AgentSpace makes professional guidance accessible to everyone — no appointments, no waiting rooms, no hourly fees.</p>
          </div>
          <div className="steps">
            <div className="step reveal">
              <div className="step-num">01</div>
              <div className="step-title">Browse the marketplace</div>
              <div className="step-body">Explore agents across therapy, law, finance, medicine, career coaching, nutrition, and education. Read detailed profiles, reviews, and what each agent specializes in.</div>
            </div>
            <div className="step reveal reveal-delay-1">
              <div className="step-num">02</div>
              <div className="step-title">Subscribe for unlimited access</div>
              <div className="step-body">Choose your expert and subscribe monthly. Get unlimited conversations for a flat fee — no per-session charges, no hidden costs. Cancel anytime.</div>
            </div>
            <div className="step reveal reveal-delay-2">
              <div className="step-num">03</div>
              <div className="step-title">Chat whenever you need to</div>
              <div className="step-body">Your agent is available 24/7 — responding in under two minutes. They remember your history, adapt to your needs, and give you the same quality advice every time.</div>
            </div>
          </div>
        </div>
      </section>

      {/* Agent preview */}
      <section className="agents-section">
        <div className="agent-feature-grid">
          {/* Left: card info */}
          <div className="reveal">
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 28 }}>
              <AgentIcon agent={AGENTS[0]} size={64} pfx="lhs"/>
              <div>
                <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 24, color: '#fff', lineHeight: 1.1 }}>Grant</div>
                <div style={{ color: grant.c2, fontSize: 13, fontWeight: 600, marginTop: 3 }}>Writing Tutor · Essays & Composition</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 5 }}>
                  <span style={{ color: '#fff', fontSize: 12, letterSpacing: 1 }}>★★★★★</span>
                  <span style={{ color: 'rgba(255,255,255,0.45)', fontSize: 12 }}>4.9 · $19/mo</span>
                </div>
              </div>
            </div>
            <div className="section-eyebrow" style={{ color: 'rgba(255,255,255,0.4)', letterSpacing: '1.8px' }}>Meet your writing tutor</div>
            <h2 className="section-title" style={{ color: '#fff', marginTop: 14 }}>Real writing expertise,<br/><em>built into always-on AI</em></h2>
            <p className="section-sub" style={{ color: 'rgba(255,255,255,0.45)', marginTop: 14 }}>Grant is built from real teaching knowledge — adaptive, honest, and available whenever you need feedback on your writing.</p>

            <div style={{ marginTop: 32, display: 'flex', flexDirection: 'column', gap: 12 }}>
              {AGENTS[0].canHelp.map((item, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                  <div style={{ width: 22, height: 22, borderRadius: 11, background: `${grant.c1}30`, border: `1.5px solid ${grant.c1}60`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: 1 }}>
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><polyline points="1.5,5.5 4,7.5 8.5,2.5" stroke={grant.c2} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  </div>
                  <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: 15, lineHeight: 1.5 }}>{item}</span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 36 }}>
              <Link to="/browse" className="btn-primary-dark">Meet Grant →</Link>
            </div>
          </div>

          {/* Right: demo placeholder */}
          <div className="reveal" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {/* Chat window */}
            <div style={{ background: '#f7f5f0', borderRadius: 20, overflow: 'hidden', boxShadow: '0 24px 64px rgba(0,0,0,0.4)' }}>
              {/* Window chrome */}
              <div style={{ background: '#1c1916', padding: '14px 18px', display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ display: 'flex', gap: 6 }}>
                  {['#ff5f57','#febc2e','#28c840'].map(c => <div key={c} style={{ width: 12, height: 12, borderRadius: 6, background: c }}/>)}
                </div>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 10 }}>
                  <AgentIcon agent={AGENTS[0]} size={28} pfx="demo"/>
                  <span style={{ color: '#fff', fontSize: 13, fontWeight: 600 }}>Grant</span>
                  <span style={{ width: 7, height: 7, borderRadius: 4, background: '#22d3ee', display: 'inline-block' }}/>
                  <span style={{ color: 'rgba(255,255,255,0.4)', fontSize: 12 }}>Online</span>
                </div>
              </div>
              {/* Messages */}
              <div style={{ padding: '20px 18px', display: 'flex', flexDirection: 'column', gap: 14, background: '#f7f5f0' }}>
                <div style={{ textAlign: 'center' }}>
                  <span style={{ background: '#eceae4', color: '#b0a89e', fontSize: 11, fontWeight: 600, padding: '3px 12px', borderRadius: 20 }}>Today</span>
                </div>
                {/* User message */}
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <div style={{ background: '#1c1916', color: '#f7f5f0', padding: '10px 14px', borderRadius: '18px 18px 4px 18px', fontSize: 14, lineHeight: 1.5, maxWidth: '80%' }}>
                    My essay intro feels weak but I'm not sure why. Here it is: "Social media has many effects on society. This essay will discuss both the positive and negative impacts."
                  </div>
                </div>
                {/* Grant message */}
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
                  <AgentIcon agent={AGENTS[0]} size={28} pfx="demo2"/>
                  <div style={{ background: '#fff', border: '1px solid #e4e1da', color: '#4a4540', padding: '10px 14px', borderRadius: '4px 18px 18px 18px', fontSize: 14, lineHeight: 1.6, maxWidth: '80%' }}>
                    Two issues. First, <strong style={{ color: '#1c1916' }}>"has many effects"</strong> tells the reader nothing — every topic has many effects.<br/><br/>
                    Second, <strong style={{ color: '#1c1916' }}>"this essay will discuss"</strong> is throat-clearing. Cut it and just make the argument.<br/><br/>
                    Try: <em style={{ color: '#1c1916' }}>"Social media hasn't just changed how we communicate — it's reshaped what we think is worth saying."</em> Now you have a real thesis. Want to refine it?
                  </div>
                </div>
                {/* User message 2 */}
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <div style={{ background: '#1c1916', color: '#f7f5f0', padding: '10px 14px', borderRadius: '18px 18px 4px 18px', fontSize: 14, lineHeight: 1.5, maxWidth: '80%' }}>
                    Yes — and can you show me how to add a hook before it?
                  </div>
                </div>
                {/* Typing indicator */}
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
                  <AgentIcon agent={AGENTS[0]} size={28} pfx="demo3"/>
                  <div style={{ background: '#fff', border: '1px solid #e4e1da', padding: '12px 16px', borderRadius: '4px 18px 18px 18px', display: 'flex', gap: 5 }}>
                    {[0,1,2].map(i => (
                      <div key={i} style={{ width: 7, height: 7, borderRadius: 4, background: '#b0a89e', animation: `bounce 1.1s ${i * 0.18}s ease-in-out infinite` }}/>
                    ))}
                  </div>
                </div>
              </div>
              {/* Input bar */}
              <div style={{ padding: '10px 14px', background: '#fff', borderTop: '1px solid #e4e1da', display: 'flex', gap: 10, alignItems: 'center' }}>
                <div style={{ flex: 1, background: '#f7f5f0', border: '1.5px solid #e4e1da', borderRadius: 20, padding: '9px 16px', fontSize: 13, color: '#b0a89e' }}>Message Grant…</div>
                <div style={{ width: 36, height: 36, borderRadius: 18, background: '#1c1916', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: 16, flexShrink: 0 }}>→</div>
              </div>
            </div>
            <div style={{ textAlign: 'center', marginTop: 14, color: 'rgba(255,255,255,0.25)', fontSize: 12, letterSpacing: '0.5px' }}>INTERACTIVE DEMO COMING SOON</div>
          </div>
        </div>
      </section>

      {/* Trust quotes */}
      <section className="trust-section">
        <div className="container">
          <div className="reveal">
            <div className="section-eyebrow">What people are saying</div>
            <h2 className="section-title">Real results,<br/><em>real people</em></h2>
          </div>
          <div className="trust-quotes">
            <div className="trust-quote reveal">
              <div className="trust-quote-text">"I submitted my college essay to 12 schools and got into my top choice. Grant helped me find the real story I was trying to tell and cut everything that wasn't it."</div>
              <div className="trust-quote-author">
                <div className="trust-quote-avatar" style={{ background: 'linear-gradient(135deg,#0891b2,#22d3ee)' }}>S</div>
                <div><div className="trust-quote-name">Sophie L.</div><div className="trust-quote-role">College applicant, Personal statement</div></div>
              </div>
            </div>
            <div className="trust-quote reveal reveal-delay-1">
              <div className="trust-quote-text">"Used Grant to prep for the AP Language exam. The feedback on my rhetorical analysis essays was more detailed than anything my teacher gave me. Scored a 5."</div>
              <div className="trust-quote-author">
                <div className="trust-quote-avatar" style={{ background: 'linear-gradient(135deg,#0891b2,#22d3ee)' }}>O</div>
                <div><div className="trust-quote-name">Omar A.</div><div className="trust-quote-role">High school senior, AP Language</div></div>
              </div>
            </div>
            <div className="trust-quote reveal reveal-delay-2">
              <div className="trust-quote-text">"My daughter's English grade went from a C to an A- in one semester. Grant doesn't just fix her writing — it explains why each change makes it better. She's actually learning."</div>
              <div className="trust-quote-author">
                <div className="trust-quote-avatar" style={{ background: 'linear-gradient(135deg,#0891b2,#22d3ee)' }}>C</div>
                <div><div className="trust-quote-name">Chloe W.</div><div className="trust-quote-role">Parent, High school English</div></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* For professionals */}
      <section className="pro-section" id="for-professionals">
        <div className="container">
          <div className="pro-inner">
            <div>
              <div className="reveal">
                <div className="section-eyebrow">For professionals</div>
                <h2 className="section-title">Turn your expertise<br/>into an AI agent.</h2>
                <p className="section-sub">We work with professionals across every field to build an AI agent from your knowledge, methods, and voice. Your expertise — available to thousands of clients, 24/7.</p>
              </div>
              <div className="pro-benefits reveal">
                <div className="pro-benefit">
                  <div className="pro-benefit-icon"><svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#1c1916" strokeWidth="1.8" strokeLinecap="round"><path d="M3 9l4 4 8-8"/></svg></div>
                  <div><div className="pro-benefit-title">Passive subscription income</div><div className="pro-benefit-body">Earn every month from every subscriber — while you sleep, travel, or work with other clients.</div></div>
                </div>
                <div className="pro-benefit">
                  <div className="pro-benefit-icon"><svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#1c1916" strokeWidth="1.8" strokeLinecap="round"><circle cx="9" cy="9" r="7"/><path d="M9 5v4l3 2"/></svg></div>
                  <div><div className="pro-benefit-title">Scale without limits</div><div className="pro-benefit-body">Serve hundreds of clients simultaneously with consistent quality — no more scheduling constraints.</div></div>
                </div>
                <div className="pro-benefit">
                  <div className="pro-benefit-icon"><svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#1c1916" strokeWidth="1.8" strokeLinecap="round"><rect x="2" y="3" width="14" height="12" rx="2"/><path d="M6 7h6M6 11h4"/></svg></div>
                  <div><div className="pro-benefit-title">You stay in control</div><div className="pro-benefit-body">Your agent reflects your methods and values. We build it with you, and you review everything before launch.</div></div>
                </div>
                <div className="pro-benefit">
                  <div className="pro-benefit-icon"><svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#1c1916" strokeWidth="1.8" strokeLinecap="round"><path d="M9 2l2.5 5 5.5.8-4 3.9.9 5.5L9 14.5l-4.9 2.7.9-5.5L1 7.8l5.5-.8z"/></svg></div>
                  <div><div className="pro-benefit-title">Launch in weeks</div><div className="pro-benefit-body">Our team handles the technical work. You provide the knowledge — we handle the rest. Most agents go live within 3–4 weeks.</div></div>
                </div>
              </div>
            </div>

            {/* Form */}
            <div className="reveal">
              <div className="apply-form">
                <h3>Apply to become an agent</h3>
                <p>Tell us about yourself and your area of expertise. We'll be in touch within 2 business days.</p>
                {!submitted ? (
                  <div>
                    <div className="form-row">
                      <div className="form-group">
                        <label>First name</label>
                        <input type="text" placeholder="Jane" value={formState.first} onChange={e => setFormState(s => ({ ...s, first: e.target.value }))} style={formError.first ? { borderColor: '#e63c2f' } : {}}/>
                      </div>
                      <div className="form-group">
                        <label>Last name</label>
                        <input type="text" placeholder="Doe" value={formState.last} onChange={e => setFormState(s => ({ ...s, last: e.target.value }))}/>
                      </div>
                    </div>
                    <div className="form-group">
                      <label>Email address</label>
                      <input type="email" placeholder="jane@example.com" value={formState.email} onChange={e => setFormState(s => ({ ...s, email: e.target.value }))} style={formError.email ? { borderColor: '#e63c2f' } : {}}/>
                    </div>
                    <div className="form-group">
                      <label>Your profession</label>
                      <select value={formState.profession} onChange={e => setFormState(s => ({ ...s, profession: e.target.value }))} style={formError.profession ? { borderColor: '#e63c2f' } : {}}>
                        <option value="" disabled>Select your field</option>
                        <option>Therapist / Psychologist</option>
                        <option>Lawyer / Attorney</option>
                        <option>Financial Advisor / CFA</option>
                        <option>Doctor / Physician</option>
                        <option>Career Coach</option>
                        <option>Nutritionist / Dietitian</option>
                        <option>Tutor / Educator</option>
                        <option>Life Coach</option>
                        <option>Other</option>
                      </select>
                    </div>
                    <div className="form-group">
                      <label>Your specialty or niche</label>
                      <input type="text" placeholder="e.g. Anxiety & CBT, Contract law, Portfolio management…" value={formState.spec} onChange={e => setFormState(s => ({ ...s, spec: e.target.value }))}/>
                    </div>
                    <div className="form-group">
                      <label>Tell us about yourself <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(optional)</span></label>
                      <textarea rows={4} placeholder="Credentials, years of experience, what makes your approach unique…" value={formState.bio} onChange={e => setFormState(s => ({ ...s, bio: e.target.value }))}/>
                    </div>
                    <button className="form-submit" type="button" onClick={handleSubmit}>Apply Now →</button>
                    <p style={{ textAlign: 'center', color: 'var(--faint)', fontSize: 12, marginTop: 12 }}>We review every application personally. No spam, ever.</p>
                  </div>
                ) : (
                  <div className="form-success visible">
                    <div className="form-success-icon">
                      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="5,14 11,20 23,8"/></svg>
                    </div>
                    <h4>Application received!</h4>
                    <p>We'll review your profile and get back to you within 2 business days. We're excited to potentially work with you.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer>
        <div className="footer-inner">
          <div>
            <div className="footer-brand">AgentSpace</div>
            <div className="footer-tagline">An AI writing tutor built from real teaching expertise. Essays, research papers, personal statements — whenever you need feedback.</div>
          </div>
          <div className="footer-col">
            <h4>Product</h4>
            <Link to="/browse">Browse Agents</Link>
            <a href="#how-it-works">How it works</a>
            <a href="#for-professionals">For professionals</a>
          </div>
        </div>
        <div className="footer-bottom">
          <span>© 2026 AgentSpace. All rights reserved.</span>
          <span>Built with care for people who deserve expert guidance.</span>
        </div>
      </footer>
    </>
  )
}
