import React from 'react';
import { Box } from '@mui/material';
import { ReactFlowProvider } from 'reactflow';
import Canvas from './Canvas';
import Sidebar from './Sidebar';

const EditorLayout = () => {
  return (
    <Box sx={{ display: 'flex', height: '100vh', width: '100vw' }}>
      <ReactFlowProvider>
        <Sidebar />
        <Canvas />
      </ReactFlowProvider>
    </Box>
  );
};

export default EditorLayout;
