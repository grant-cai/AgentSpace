const shapes = {
  1:   <><circle cx="22" cy="26" r="8" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5"/><circle cx="34" cy="26" r="8" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5"/></>,
  2:   <><path d="M16 36L28 18L40 36" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"/><line x1="21" y1="36" x2="35" y2="36" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5" strokeLinecap="round"/></>,
  3:   <><rect x="13" y="31" width="6" height="9" rx="1.5" fill="rgba(255,255,255,0.85)"/><rect x="22" y="24" width="6" height="16" rx="1.5" fill="rgba(255,255,255,0.85)"/><rect x="31" y="18" width="6" height="22" rx="1.5" fill="rgba(255,255,255,0.85)"/></>,
  4:   <path d="M28 16L32 25L42 27L35 34L37 44L28 40L19 44L21 34L14 27L24 25Z" fill="rgba(255,255,255,0.85)"/>,
  5:   <><circle cx="28" cy="24" r="8" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5"/><line x1="28" y1="32" x2="28" y2="40" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5" strokeLinecap="round"/><line x1="21" y1="40" x2="35" y2="40" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5" strokeLinecap="round"/></>,
  6:   <path d="M28 16C22 18 14 24 15 31C16 38 22 41 28 41C34 41 37 37 37 32C37 27 32 26 30 29C28 32 22 30 28 16Z" fill="rgba(255,255,255,0.85)"/>,
  7:   <><line x1="28" y1="17" x2="28" y2="39" stroke="rgba(255,255,255,0.85)" strokeWidth="3" strokeLinecap="round"/><line x1="17" y1="28" x2="39" y2="28" stroke="rgba(255,255,255,0.85)" strokeWidth="3" strokeLinecap="round"/></>,
  101: <><circle cx="22" cy="26" r="8" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5"/><circle cx="34" cy="26" r="8" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5"/></>,
  102: <><path d="M16 36L28 18L40 36" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"/><line x1="21" y1="36" x2="35" y2="36" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5" strokeLinecap="round"/></>,
  103: <><rect x="13" y="31" width="6" height="9" rx="1.5" fill="rgba(255,255,255,0.85)"/><rect x="22" y="24" width="6" height="16" rx="1.5" fill="rgba(255,255,255,0.85)"/><rect x="31" y="18" width="6" height="22" rx="1.5" fill="rgba(255,255,255,0.85)"/></>,
}

export default function AgentIcon({ agent, size = 52, pfx = 'g' }) {
  if (agent.img) {
    return (
      <img
        src={agent.img}
        alt={agent.name}
        width={size}
        height={size}
        style={{ flexShrink: 0, borderRadius: `${size * 0.25}px`, objectFit: 'cover', display: 'block' }}
      />
    )
  }
  const id = `${pfx}${agent.id}`
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" style={{ flexShrink: 0 }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor={agent.c1}/>
          <stop offset="1" stopColor={agent.c2}/>
        </linearGradient>
      </defs>
      <rect width="56" height="56" rx="14" fill={`url(#${id})`}/>
      {shapes[agent.id]}
    </svg>
  )
}
