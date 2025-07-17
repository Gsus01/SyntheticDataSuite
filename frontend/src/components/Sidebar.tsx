import { Box, Card, Typography } from '@mui/material'

const blocks = ['Preprocess', 'Train HMM', 'Generate']

export default function Sidebar() {
  return (
    <Box className="w-60 p-2 overflow-y-auto border-r border-gray-200 dark:border-gray-700" component="aside">
      {blocks.map((b) => (
        <Card key={b} className="mb-2 p-2 cursor-grab">
          <Typography>{b}</Typography>
        </Card>
      ))}
    </Box>
  )
}
