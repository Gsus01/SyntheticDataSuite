import React from 'react';

const TopBar: React.FC = () => {
  return (
    <div className="bg-neutral-900 text-white p-3 flex justify-between items-center shadow-md">
      <h1 className="text-xl font-semibold">Argo Workflow Designer</h1>
      <div>
        {/* Botones y switch de modo oscuro irán aquí */}
        <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mr-2">
          Validate
        </button>
        <button className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded mr-2">
          Generate YAML
        </button>
        <button className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded">
          Run Workflow
        </button>
        {/* Dark Mode Switch Placeholder */}
      </div>
    </div>
  );
};

export default TopBar;
