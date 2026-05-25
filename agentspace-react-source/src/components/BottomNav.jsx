import { useC } from '../context/ThemeContext'

export default function BottomNav({ tab, setTab, owned }) {
  const C = useC()
  const items = [
    { id: 'browse', label: 'Browse', icon: <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><rect x="2" y="2" width="7" height="7" rx="1.5"/><rect x="11" y="2" width="7" height="7" rx="1.5"/><rect x="2" y="11" width="7" height="7" rx="1.5"/><rect x="11" y="11" width="7" height="7" rx="1.5"/></svg> },
    { id: 'myagents', label: 'My Agents', badge: owned.length, icon: <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="10" cy="7" r="3.5"/><path d="M3 18c0-3.866 3.134-7 7-7s7 3.134 7 7"/></svg> },
  ]
  return (
    <div style={{ flexShrink: 0, background: C.card, borderTop: `1px solid ${C.line}`, display: 'flex', padding: '4px 0 max(4px,env(safe-area-inset-bottom))' }}>
      {items.map(item => (
        <button key={item.id} onClick={() => setTab(item.id)}
          style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, padding: '10px 12px', border: 'none', cursor: 'pointer', background: 'transparent', color: tab === item.id ? C.ink : C.faint, fontFamily: 'inherit', fontSize: 11, fontWeight: tab === item.id ? 700 : 400, position: 'relative', transition: 'color 0.15s' }}>
          {item.icon}
          {item.label}
          {item.badge > 0 && <span style={{ position: 'absolute', top: 6, right: 'calc(50% - 14px)', background: C.accent, color: '#fff', fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 10 }}>{item.badge}</span>}
          {tab === item.id && <div style={{ position: 'absolute', bottom: 0, width: 24, height: 3, background: C.ink, borderRadius: '2px 2px 0 0' }}/>}
        </button>
      ))}
    </div>
  )
}
