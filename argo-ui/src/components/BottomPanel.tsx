import React, { useState } from 'react';
import { Tab } from '@headlessui/react';
import { ChevronDown, ChevronUp } from 'lucide-react'; // Iconos para colapsar

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

const BottomPanel: React.FC = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);

  const tabs = ['Status', 'Logs', 'Artifacts'];

  return (
    <div className={`bg-neutral-100 border-t border-neutral-300 ${isCollapsed ? 'h-12' : 'h-64'} transition-height duration-300 ease-in-out`}>
      <div className="flex justify-between items-center p-2 bg-neutral-200 cursor-pointer" onClick={() => setIsCollapsed(!isCollapsed)}>
        <Typography variant="subtitle2" className="font-medium text-neutral-700">
          {tabs[selectedTab]}
        </Typography>
        {isCollapsed ? <ChevronUp size={20} className="text-neutral-600" /> : <ChevronDown size={20} className="text-neutral-600" />}
      </div>

      {!isCollapsed && (
        <Tab.Group defaultIndex={selectedTab} onChange={setSelectedTab}>
          <Tab.List className="flex space-x-1 rounded-t-lg bg-neutral-200 p-1">
            {tabs.map((tabName) => (
              <Tab
                key={tabName}
                className={({ selected }) =>
                  classNames(
                    'w-full rounded-lg py-2 px-3 text-sm font-medium leading-5',
                    'focus:outline-none focus:ring-2 ring-offset-2 ring-offset-blue-400 ring-white ring-opacity-60',
                    selected
                      ? 'bg-white shadow text-blue-700'
                      : 'text-neutral-600 hover:bg-white/[0.12] hover:text-neutral-800'
                  )
                }
              >
                {tabName}
              </Tab>
            ))}
          </Tab.List>
          <Tab.Panels className="h-full p-3 overflow-y-auto bg-white">
            <Tab.Panel>
              <Typography className="text-neutral-600">Status Panel Content - Tree view of nodes will be here.</Typography>
            </Tab.Panel>
            <Tab.Panel>
              <Typography className="text-neutral-600">Logs Panel Content - Log stream will appear here.</Typography>
            </Tab.Panel>
            <Tab.Panel>
              <Typography className="text-neutral-600">Artifacts Panel Content - Links to S3 artifacts.</Typography>
            </Tab.Panel>
          </Tab.Panels>
        </Tab.Group>
      )}
    </div>
  );
};

import { Typography } from '@mui/material'; // Importar Typography de MUI

export default BottomPanel;
