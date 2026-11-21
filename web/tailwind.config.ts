import type { Config } from "tailwindcss";

// Every colour is a CSS variable (index.css) holding "r g b" channels, so light and dark themes share one set of
// class names and Tailwind's opacity modifiers (bg-accent/10) keep working.
// Colour encodes meaning: lexical methods are cobalt, dense / semantic is violet, expansion is amber,
// a step that reached τ is teal, one that fell short is crimson.
const v = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        canvas: v("canvas"),
        surface: v("surface"),
        raised: v("raised"),
        sunken: v("sunken"),
        ink: { DEFAULT: v("ink"), soft: v("ink-soft"), faint: v("ink-faint") },
        line: v("line"),
        accent: { DEFAULT: v("accent"), soft: v("accent-soft"), ink: v("accent-ink") },
        lexical: v("lexical"),
        semantic: { DEFAULT: v("semantic"), soft: v("semantic-soft") },
        expand: { DEFAULT: v("expand"), soft: v("expand-soft") },
        pass: { DEFAULT: v("pass"), soft: v("pass-soft") },
        fail: { DEFAULT: v("fail"), soft: v("fail-soft") },
      },
      fontFamily: {
        sans: ['"Instrument Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ['"Newsreader"', "ui-serif", "Georgia", "serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      borderRadius: { panel: "14px", ctl: "9px" },
      boxShadow: {
        panel: "0 1px 0 rgb(var(--shadow) / 0.04), 0 1px 3px rgb(var(--shadow) / 0.06)",
        lift: "0 1px 2px rgb(var(--shadow) / 0.06), 0 12px 32px -12px rgb(var(--shadow) / 0.28)",
        focus: "0 0 0 3px rgb(var(--accent) / 0.25)",
      },
      transitionTimingFunction: { out: "cubic-bezier(0.2, 0.8, 0.2, 1)" },
      keyframes: {
        reveal: { from: { opacity: "0", transform: "translateY(6px)" }, to: { opacity: "1", transform: "none" } },
        grow: { from: { transform: "scaleX(0)" }, to: { transform: "scaleX(1)" } },
        shimmer: { "0%": { backgroundPosition: "-480px 0" }, "100%": { backgroundPosition: "480px 0" } },
      },
      animation: {
        reveal: "reveal 420ms cubic-bezier(0.2, 0.8, 0.2, 1) both",
        grow: "grow 700ms cubic-bezier(0.2, 0.8, 0.2, 1) both",
        shimmer: "shimmer 1.3s linear infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
