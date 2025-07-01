import React from 'react';
import { Drawer, Typography, Box, IconButton } from '@mui/material';
import { X } from 'lucide-react'; // Icono para cerrar

interface InspectorProps {
  isOpen: boolean;
  onClose: () => void;
  selectedNodeConfig: any; // Debería ser un tipo más específico basado en Zod schema
}

const Inspector: React.FC<InspectorProps> = ({ isOpen, onClose, selectedNodeConfig }) => {
  return (
    <Drawer
      anchor="right"
      open={isOpen}
      onClose={onClose}
      PaperProps={{
        sx: { width: 360, backgroundColor: 'neutral.50', borderLeft: '1px solid', borderColor: 'neutral.200' }
      }}
    >
      <Box className="p-4">
        <Box className="flex justify-between items-center mb-4">
          <Typography variant="h6" className="text-neutral-700">Node Configuration</Typography>
          <IconButton onClick={onClose} size="small">
            <X size={20} className="text-neutral-500" />
          </IconButton>
        </Box>

        {selectedNodeConfig ? (
          <Box>
            <Typography variant="subtitle1">Node ID: {selectedNodeConfig.id}</Typography>
            <Typography variant="subtitle1">Type: {selectedNodeConfig.type}</Typography>
            {/* Aquí se generará el formulario dinámicamente */}
            <Typography className="mt-4 text-neutral-600">
              Form fields for '{selectedNodeConfig.data?.label || selectedNodeConfig.type}' will appear here.
            </Typography>
          </Box>
        ) : (
          <Typography className="text-neutral-500">Select a node to configure its parameters.</Typography>
        )}
      </Box>
    </Drawer>
  );
};

export default Inspector;
