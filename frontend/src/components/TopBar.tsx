import { AppBar, Toolbar, Button, Switch, Typography } from '@mui/material'

interface Props {
  onToggleDark: () => void
}

export default function TopBar({ onToggleDark }: Props) {
  return (
    <AppBar position="static" color="default" sx={{ borderBottom: 1, borderColor: 'divider' }}>
      <Toolbar variant="dense">
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Argo Designer
        </Typography>
        <Button color="inherit">Validate</Button>
        <Button color="inherit">Generate YAML</Button>
        <Button color="inherit">Run Workflow</Button>
        <Switch onChange={onToggleDark} />
      </Toolbar>
    </AppBar>
  )
}
