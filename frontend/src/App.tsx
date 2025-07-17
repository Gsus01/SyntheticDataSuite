import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import { useMemo, useState } from 'react'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import BottomPanel from './components/BottomPanel'
import WorkflowCanvas from './components/WorkflowCanvas'
import Inspector from './components/Inspector'

function App() {
  const [dark, setDark] = useState(false)
  const theme = useMemo(() => createTheme({ palette: { mode: dark ? 'dark' : 'light' } }), [dark])

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="flex flex-col h-screen">
        <TopBar onToggleDark={() => setDark(!dark)} />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <WorkflowCanvas />
          <Inspector />
        </div>
        <BottomPanel />
      </div>
    </ThemeProvider>
  )
}

export default App
