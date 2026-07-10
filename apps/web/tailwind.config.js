/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{html,ts}'],
  theme: {
    extend: {
      colors: {
        app: '#f5f7fb',
        surface: '#ffffff',
        'surface-muted': '#f8fafc',
        border: '#dbe3ef',
        'border-strong': '#b9c5d6',
        text: '#172033',
        'text-muted': '#64748b',
        primary: '#3157d5',
        'primary-strong': '#2447b8',
        'primary-soft': '#e7ecff',
        success: '#147d52',
        warning: '#a15c00',
        danger: '#b42318',
      },
      boxShadow: {
        app: '0 18px 45px rgb(23 32 51 / 8%)',
        soft: '0 10px 28px rgb(23 32 51 / 6%)',
      },
      spacing: {
        4.5: '1.125rem',
        5.5: '1.375rem',
      },
    },
  },
  plugins: [],
};
