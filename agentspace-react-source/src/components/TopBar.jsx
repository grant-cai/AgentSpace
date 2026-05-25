import { Link } from 'react-router-dom'
import TabToggle from './TabToggle'
import { useTheme } from '../context/ThemeContext'

export default function TopBar({ tab, owned, setTab }) {
  const { C } = useTheme()
  return (
    <div style={{ position: 'sticky', top: 0, zIndex: 50, background: C.card, borderBottom: `1px solid ${C.line}`, padding: '0 20px', height: 56, display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', flexShrink: 0 }}>
      <Link to="/" style={{ fontFamily: "'DM Serif Display',serif", fontSize: 21, letterSpacing: '-0.3px', whiteSpace: 'nowrap', textDecoration: 'none', color: C.ink }}>
        AgentSpace
      </Link>
      <TabToggle tab={tab} setTab={setTab} owned={owned}/>
      <div/>
    </div>
  )
}
