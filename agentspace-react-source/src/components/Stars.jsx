export default function Stars({ n = 5 }) {
  return (
    <span style={{ color: '#d97706', letterSpacing: 1, fontSize: 13 }}>
      {'★'.repeat(Math.floor(n))}
      <span style={{ opacity: 0.22 }}>{'★'.repeat(5 - Math.floor(n))}</span>
    </span>
  )
}
