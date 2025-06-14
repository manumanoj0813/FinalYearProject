import { extendTheme, type ThemeConfig } from '@chakra-ui/react';

const config: ThemeConfig = {
  initialColorMode: 'light',
  useSystemColorMode: false,
};

export const theme = extendTheme({
  config,
  colors: {
    brand: {
      50: '#f0f9ff',
      100: '#e0f2fe',
      200: '#bae6fd',
      300: '#7dd3fc',
      400: '#38bdf8',
      500: '#0ea5e9',
      600: '#0284c7',
      700: '#0369a1',
      800: '#075985',
      900: '#0c4a6e',
    },
    accent: {
      50: '#fdf4ff',
      100: '#fae8ff',
      200: '#f5d0fe',
      300: '#f0abfc',
      400: '#e879f9',
      500: '#d946ef',
      600: '#c026d3',
      700: '#a21caf',
      800: '#86198f',
      900: '#701a75',
    },
    gradient: {
      primary: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      secondary: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      ocean: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      sunset: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
      aurora: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
      cosmic: 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%)',
    },
  },
  fonts: {
    heading: '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif',
    body: '"Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, sans-serif',
  },
  components: {
    Button: {
      baseStyle: {
        fontWeight: 'semibold',
        borderRadius: 'lg',
        _focus: {
          boxShadow: '0 0 0 3px rgba(14, 165, 233, 0.3)',
        },
      },
      variants: {
        solid: {
          bg: 'gradient.primary',
          color: 'white',
          _hover: {
            bg: 'gradient.secondary',
            transform: 'translateY(-2px)',
            boxShadow: 'lg',
          },
          _active: {
            transform: 'translateY(0)',
          },
        },
        outline: {
          bg: 'transparent',
          color: 'brand.500',
          border: '2px solid',
          borderColor: 'brand.500',
          _hover: {
            bg: 'brand.50',
            borderColor: 'brand.600',
            color: 'brand.600',
          },
        },
      },
      defaultProps: {
        variant: 'solid',
        colorScheme: 'brand',
      },
    },
    Card: {
      baseStyle: {
        container: {
          bg: 'white',
          border: '1px solid',
          borderColor: 'gray.200',
          borderRadius: 'xl',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          _hover: {
            transform: 'translateY(-2px)',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          },
          transition: 'all 0.3s ease',
        },
      },
    },
    Progress: {
      baseStyle: {
        track: {
          bg: 'gray.100',
          borderRadius: 'full',
        },
        filledTrack: {
          bg: 'gradient.ocean',
          borderRadius: 'full',
        },
      },
      defaultProps: {
        colorScheme: 'brand',
      },
    },
    Tabs: {
      variants: {
        'soft-rounded': {
          tab: {
            bg: 'gray.50',
            color: 'gray.600',
            fontWeight: 'medium',
            _selected: {
              bg: 'gradient.primary',
              color: 'white',
              boxShadow: 'md',
            },
            _hover: {
              bg: 'gray.100',
            },
          },
        },
      },
    },
  },
  styles: {
    global: (props: any) => ({
      'html, body': {
        scrollBehavior: 'smooth',
      },
      body: {
        bg: props.colorMode === 'dark' ? 'gray.900' : '#f8fafc',
        color: props.colorMode === 'dark' ? 'white' : 'gray.800',
        fontFamily: 'body',
        lineHeight: 'base',
        minHeight: '100vh',
      },
      '*': {
        boxSizing: 'border-box',
      },
      '::selection': {
        bg: 'brand.200',
        color: 'brand.800',
      },
      '::-webkit-scrollbar': {
        width: '8px',
      },
      '::-webkit-scrollbar-track': {
        bg: 'gray.100',
        borderRadius: '4px',
      },
      '::-webkit-scrollbar-thumb': {
        bg: 'brand.300',
        borderRadius: '4px',
        '&:hover': {
          bg: 'brand.400',
        },
      },
    }),
  },
}); 