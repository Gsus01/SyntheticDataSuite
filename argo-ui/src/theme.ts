import { createTheme } from '@mui/material/styles';
import { red } from '@mui/material/colors';

// Un tema básico de ejemplo para la aplicación Argo UI
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2', // Un azul estándar de MUI
    },
    secondary: {
      main: '#dc004e', // Un rosa estándar de MUI
    },
    error: {
      main: red.A400,
    },
    background: {
      default: '#f4f6f8', // Un gris claro para el fondo general
      paper: '#ffffff',   // Blanco para superficies como Cards, Drawers
    },
    text: {
      primary: '#2c3e50', // Un color de texto principal más oscuro y suave
      secondary: '#7f8c8d', // Un color de texto secundario
    }
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(','),
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff', // Ejemplo: AppBar blanco
          color: '#2c3e50',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#ffffff', // Asegurar que los drawers usen el color de paper
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8, // Bordes ligeramente más redondeados para las cards
          boxShadow: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)', // Sombra sutil
        }
      }
    }
    // Aquí se pueden añadir más overrides de componentes
  }
});

export default theme;
