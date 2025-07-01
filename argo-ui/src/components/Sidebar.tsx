import React from 'react';
import { Card, CardContent, Typography, TextField } from '@mui/material';

// Iconos de ejemplo (lucide-react)
import { PlayCircle, Settings, Zap } from 'lucide-react';

interface NodePaletteItem {
  id: string;
  type: string;
  name: string;
  icon: React.ElementType;
  tags: string[];
  category: string;
}

const paletteItems: NodePaletteItem[] = [
  { id: 'preprocess', type: 'preprocess', name: 'Preprocess Data', icon: Settings, tags: ['data', 'clean'], category: 'preprocessing' },
  { id: 'train-hmm', type: 'trainHMM', name: 'Train HMM', icon: PlayCircle, tags: ['model', 'train'], category: 'training' },
  { id: 'generate-data', type: 'generateData', name: 'Generate Data', icon: Zap, tags: ['synthesis', 'create'], category: 'generation' },
];

const Sidebar: React.FC = () => {
  const onDragStart = (event: React.DragEvent, nodeType: string, nodeName: string) => {
    event.dataTransfer.setData('application/reactflow-type', nodeType);
    event.dataTransfer.setData('application/reactflow-name', nodeName);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="w-72 bg-neutral-50 border-r border-neutral-200 p-4 space-y-4 overflow-y-auto">
      <Typography variant="h6" className="mb-4 text-neutral-700">Palette</Typography>
      <TextField label="Search nodes..." variant="outlined" size="small" fullWidth className="mb-4" />
      {/* Aquí irían los filtros por categoría */}

      {paletteItems.map((item) => {
        const Icon = item.icon;
        return (
          <Card
            key={item.id}
            className="mb-2 cursor-grab shadow-sm hover:shadow-md transition-shadow"
            onDragStart={(event) => onDragStart(event, item.type, item.name)}
            draggable
          >
            <CardContent className="flex items-center p-3">
              <Icon className="mr-3 text-blue-500" size={24} />
              <Typography variant="subtitle1" className="text-neutral-800">{item.name}</Typography>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};

export default Sidebar;
