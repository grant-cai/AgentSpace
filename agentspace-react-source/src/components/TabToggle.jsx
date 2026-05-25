import { useC } from '../context/ThemeContext'

export default function TabToggle({ tab, setTab, owned }) {
  const C = useC()
  const tabs = [{ id: 'browse', label: 'Marketplace' }, { id: 'myagents', label: 'My Agents' }]
  return (
    <div style={{ display: 'inline-flex', background: C.subtle, borderRadius: 11, padding: 3, gap: 2 }}>
      {tabs.map(t => (
        <button key={t.id} onClick={() => setTab(t.id)}
          style={{ padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontWeight: tab === t.id ? 700 : 500, fontSize: 14, background: tab === t.id ? C.card : 'transparent', color: tab === t.id ? C.ink : C.muted, transition: 'all 0.15s', boxShadow: tab === t.id ? '0 1px 4px rgba(0,0,0,0.08)' : 'none', position: 'relative', display: 'flex', alignItems: 'center', gap: 7 }}>
          {t.label}
          {t.id === 'myagents' && owned.length > 0 && (
            <span style={{ background: C.accent, color: '#fff', fontSize: 10, fontWeight: 700, padding: '1px 6px', borderRadius: 10, lineHeight: 1.6 }}>{owned.length}</span>
          )}
        </button>
      ))}
    </div>
  )
}
