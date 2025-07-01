import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from './theme'; // Importar el tema personalizado
import App from './App.tsx';
import './index.css'; // Tailwind CSS
import 'reactflow/dist/style.css'; // React Flow styles

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* Normaliza estilos y aplica fondo del tema */}
      <App />
    </ThemeProvider>
  </StrictMode>,
);
