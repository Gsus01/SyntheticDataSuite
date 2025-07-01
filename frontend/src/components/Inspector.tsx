import { Drawer, Box, Typography } from '@mui/material'

export default function Inspector() {
  return (
    <Drawer variant="persistent" anchor="right" open={false} hideBackdrop>
      <Box className="w-72 p-4">
        <Typography variant="h6">Inspector</Typography>
      </Box>
    </Drawer>
  )
}
