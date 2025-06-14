import { extendTheme, type ThemeConfig } from '@chakra-ui/react';

const config: ThemeConfig = {
  initialColorMode: 'light',
  useSystemColorMode: false,
};

export const theme = extendTheme({
  config,
  colors: {
    brand: {
      50: '#f5e9ff',
      100: '#dbc1ff',
      200: '#c199ff',
      300: '#a770ff',
      400: '#8d48ff',
      500: '#741fff',
      600: '#5a15cc',
      700: '#410c99',
      800: '#280466',
      900: '#100033',
    },
  },
  fonts: {
    heading: '"Inter", sans-serif',
    body: '"Inter", sans-serif',
  },
  components: {
    Button: {
      defaultProps: {
        colorScheme: 'purple',
      },
    },
    Progress: {
      defaultProps: {
        colorScheme: 'purple',
      },
    },
  },
  styles: {
    global: (props: any) => ({
      body: {
        bg: props.colorMode === 'dark' ? 'gray.900' : 'gray.50',
        color: props.colorMode === 'dark' ? 'white' : 'gray.800',
      },
    }),
  },
}); 