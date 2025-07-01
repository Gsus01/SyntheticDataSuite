import React, { useState } from 'react';
import TopBar from '../components/TopBar';
import Sidebar from '../components/Sidebar';
import Canvas from '../components/Canvas';
import Inspector from '../components/Inspector';
import BottomPanel from '../components/BottomPanel';
import { ReactFlowProvider } from 'reactflow';

const DesignerPage: React.FC = () => {
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [selectedNodeConfig, setSelectedNodeConfig] = useState<any>(null);

  // Mock function to open inspector - en el futuro se llamará al seleccionar un nodo
  const handleNodeSelect = (nodeConfig: any) => {
    setSelectedNodeConfig(nodeConfig);
    setIsInspectorOpen(true);
  };

  const handleInspectorClose = () => {
    setIsInspectorOpen(false);
    setSelectedNodeConfig(null);
  };

  // Mock: Simular selección de nodo para probar el inspector
  React.useEffect(() => {
    // handleNodeSelect({ id: '1', type: 'input', data: { label: 'Start Node' } });
  }, []);

  return (
    <ReactFlowProvider> {/* Necesario para que los hooks de React Flow funcionen en componentes hijos */}
      <div className="flex flex-col h-screen bg-neutral-50 text-neutral-800">
        <TopBar />

        <div className="flex flex-1 overflow-hidden">
          <Sidebar />

          {/* Canvas necesita un contenedor que defina su tamaño */}
          <div className="flex-1 relative"> {/* Added relative for potential absolute positioned elements within Canvas */}
            <Canvas /> {/* Canvas ahora debería tomar el espacio disponible */}
          </div>

          <Inspector
            isOpen={isInspectorOpen}
            onClose={handleInspectorClose}
            selectedNodeConfig={selectedNodeConfig}
          />
        </div>

        <BottomPanel />
      </div>
    </ReactFlowProvider>
  );
};

export default DesignerPage;
