/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#eef2ff',
          100: '#e0e7ff',
          400: '#818cf8',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
          900: '#1e1b4b',
        },
        genuine: {
          50:  '#f0fdf4',
          400: '#4ade80',
          600: '#16a34a',
        },
        fake: {
          50:  '#fff1f2',
          400: '#f87171',
          600: '#dc2626',
        },
        suspicious: {
          50:  '#fffbeb',
          400: '#fbbf24',
          600: '#d97706',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan': 'scan 2s ease-in-out infinite',
        'score-fill': 'score-fill 1.5s ease-out forwards',
      },
      keyframes: {
        scan: {
          '0%, 100%': { transform: 'translateY(0%)' },
          '50%': { transform: 'translateY(100%)' },
        },
        'score-fill': {
          from: { strokeDashoffset: '377' },
          to: { strokeDashoffset: 'var(--target-offset)' },
        },
      },
    },
  },
  plugins: [],
}
