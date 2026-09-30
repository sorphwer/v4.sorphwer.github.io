const defaultTheme = require('tailwindcss/defaultTheme')
const colors = require('tailwindcss/colors')
const { DEFAULT } = require('@tailwindcss/typography/src/styles')

// Geist tokens → Tailwind color keys. `text-ds-gray-900`, `bg-ds-background-100`,
// `border-ds-gray-alpha-400`, `text-ds-foreground` … all resolve to CSS variables
// declared in css/paper.css.
function dsPalette() {
  const scale = (name, steps) => Object.fromEntries(steps.map((s) => [s, `var(--ds-${name}-${s})`]))
  const full = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  const semantic = [
    'foreground',
    'card',
    'card-foreground',
    'popover',
    'popover-foreground',
    'primary',
    'primary-foreground',
    'secondary',
    'secondary-foreground',
    'muted',
    'muted-foreground',
    'accent',
    'accent-foreground',
    'destructive',
    'border',
    'input',
    'ring',
    'chart-1',
    'chart-2',
    'chart-3',
    'chart-4',
    'chart-5',
  ]
  return {
    background: { ...scale('background', [100, 200]), DEFAULT: 'var(--ds-sem-background)' },
    gray: { ...scale('gray', full), alpha: scale('gray-alpha', [100, 200, 300, 400, 500, 600]) },
    blue: scale('blue', full),
    red: scale('red', full),
    amber: scale('amber', full),
    green: scale('green', full),
    teal: scale('teal', [100, 300, 600, 700, 900, 1000]),
    purple: scale('purple', [100, 300, 600, 700, 900, 1000]),
    pink: scale('pink', [100, 300, 700, 900]),
    link: 'var(--ds-link)',
    ...Object.fromEntries(semantic.map((k) => [k, `var(--ds-sem-${k})`])),
  }
}

module.exports = {
  experimental: {
    optimizeUniversalDefaults: true,
  },
  content: [
    './pages/**/*.{js,jsx,ts,tsx}',
    './components/**/*.{js,jsx,ts,tsx}',
    './layouts/**/*.{js,jsx,ts,tsx}',
    './lib/**/*.{js,jsx,ts,tsx}',
    './data/**/*.mdx',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      spacing: {
        '9/16': '56.25%',
      },
      lineHeight: {
        11: '2.75rem',
        12: '3rem',
        13: '3.25rem',
        14: '3.5rem',
      },
      fontFamily: {
        sans: ['InterVariable', ...defaultTheme.fontFamily.sans],
        rs: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"',
        geist: ['"Geist Variable"', ...defaultTheme.fontFamily.sans],
        'geist-mono': ['"Geist Mono Variable"', ...defaultTheme.fontFamily.mono],
      },
      boxShadow: {
        'ds-border': 'var(--ds-shadow-border)',
        'ds-small': 'var(--ds-shadow-small)',
        'ds-medium': 'var(--ds-shadow-medium)',
        'ds-modal': 'var(--ds-shadow-modal)',
      },
      keyframes: {
        paperRise: {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'paper-rise': 'paperRise 600ms var(--ds-ease) both',
      },
      colors: {
        gray: colors.neutral,
        primary: {
          400: '#64d2ff',
          500: '#0070c9',
          600: '#0070c9',
          light: '#64d2ff',
          DEFAULT: '#0070c9',
          dark: '#0070c9',
        },
        RSpink: '#e83e8c',
        RSgrey: '#808080',
        // Geist design-system palette for the paper post; every value is a CSS
        // variable scoped under `.paper` (css/paper.css) so it flips with `.dark`
        // and never leaks into the rest of the site.
        ds: dsPalette(),
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.gray.700'),
            a: {
              color: theme('colors.primary.500'),
              '&:hover': {
                color: `${theme('colors.primary.600')} !important`,
              },
              code: { color: theme('colors.primary.400') },
            },
            h1: {
              fontWeight: '700',
              letterSpacing: theme('letterSpacing.tight'),
              color: theme('colors.gray.900'),
            },
            h2: {
              fontWeight: '700',
              letterSpacing: theme('letterSpacing.tight'),
              color: theme('colors.gray.900'),
            },
            h3: {
              fontWeight: '600',
              color: theme('colors.gray.900'),
            },
            'h4,h5,h6': {
              color: theme('colors.gray.900'),
            },
            pre: {
              backgroundColor: theme('colors.gray.800'),
            },
            code: {
              color: theme('colors.RSpink'),
              backgroundColor: theme('colors.gray.100'),
              paddingLeft: '4px',
              paddingRight: '4px',
              paddingTop: '2px',
              paddingBottom: '2px',
              borderRadius: '0.25rem',
            },
            'code::before': {
              content: 'none',
            },
            'code::after': {
              content: 'none',
            },
            details: {
              backgroundColor: theme('colors.gray.100'),
              paddingLeft: '4px',
              paddingRight: '4px',
              paddingTop: '2px',
              paddingBottom: '2px',
              borderRadius: '0.25rem',
            },
            hr: { borderColor: theme('colors.gray.200') },
            'ol li::marker': {
              fontWeight: '600',
              color: theme('colors.gray.500'),
            },
            'ul li::marker': {
              backgroundColor: theme('colors.gray.500'),
            },
            strong: { color: theme('colors.gray.600') },
            blockquote: {
              color: theme('colors.gray.900'),
              borderLeftColor: theme('colors.gray.200'),
            },
          },
        },
        dark: {
          css: {
            color: theme('colors.gray.300'),
            a: {
              color: theme('colors.primary.500'),
              '&:hover': {
                color: `${theme('colors.primary.400')} !important`,
              },
              code: { color: theme('colors.primary.400') },
            },
            h1: {
              fontWeight: '700',
              letterSpacing: theme('letterSpacing.tight'),
              color: theme('colors.gray.100'),
            },
            h2: {
              fontWeight: '700',
              letterSpacing: theme('letterSpacing.tight'),
              color: theme('colors.gray.100'),
            },
            h3: {
              fontWeight: '600',
              color: theme('colors.gray.100'),
            },
            'h4,h5,h6': {
              color: theme('colors.gray.100'),
            },
            pre: {
              backgroundColor: theme('colors.gray.800'),
            },
            code: {
              backgroundColor: theme('colors.gray.800'),
            },
            details: {
              backgroundColor: theme('colors.gray.800'),
            },
            hr: { borderColor: theme('colors.gray.700') },
            'ol li::marker': {
              fontWeight: '600',
              color: theme('colors.gray.400'),
            },
            'ul li::marker': {
              backgroundColor: theme('colors.gray.400'),
            },
            strong: { color: theme('colors.gray.100') },
            thead: {
              th: {
                color: theme('colors.gray.100'),
              },
            },
            tbody: {
              tr: {
                borderBottomColor: theme('colors.gray.700'),
              },
            },
            blockquote: {
              color: theme('colors.gray.100'),
              borderLeftColor: theme('colors.gray.700'),
            },
          },
        },
      }),
    },
    // mermaid: (theme) => ({
    //   light: {
    //     // Modify the styles for light mode here
    //     '& path': {
    //       fill: theme('colors.gray.500'),
    //     },
    //   },
    //   dark: {
    //     // Modify the styles for dark mode here
    //     '& path': {
    //       fill: theme('colors.gray.200'),
    //     },
    //   },
    // }),
  },
  plugins: [require('@tailwindcss/forms'), require('@tailwindcss/typography')],
}
