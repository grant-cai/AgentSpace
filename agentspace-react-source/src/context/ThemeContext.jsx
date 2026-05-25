import { createContext, useContext } from 'react'

export const LIGHT = {
  bg:'#f7f5f0', card:'#ffffff', ink:'#1c1916', mid:'#4a4540',
  muted:'#7a746b', faint:'#b0a89e', subtle:'#eceae4', line:'#e4e1da',
  hero:'#1c1916', accent:'#e63c2f', green:'#16a34a',
}

const ThemeContext = createContext({ C: LIGHT })

export function ThemeProvider({ children }) {
  return (
    <ThemeContext.Provider value={{ C: LIGHT }}>
      {children}
    </ThemeContext.Provider>
  )
}

export const useTheme = () => useContext(ThemeContext)
export const useC = () => useContext(ThemeContext).C
