import { create } from 'zustand';
import { getItemWithLegacy } from '../utils/legacyStorage';
import type { EmbedFontSize, SubtitleFontFamily, SubtitleStyleSettings } from '../types';

export type Language = 'en' | 'he';
export type Theme = 'dark' | 'light';
export type View = 'studio' | 'settings' | 'help';

const STORAGE_KEY = 'subbuild.ui';

export const MIN_SUBTITLE_FONT_SIZE = 12;
export const MAX_SUBTITLE_FONT_SIZE = 36;

export interface SubtitleFontOption {
  id: SubtitleFontFamily;
  stack: string;
}

export const SUBTITLE_FONTS: SubtitleFontOption[] = [
  { id: 'Heebo', stack: "'Heebo', 'Inter', system-ui, sans-serif" },
  { id: 'Rubik', stack: "'Rubik', 'Heebo', system-ui, sans-serif" },
  { id: 'Assistant', stack: "'Assistant', 'Heebo', system-ui, sans-serif" },
  { id: 'Secular One', stack: "'Secular One', 'Heebo', system-ui, sans-serif" },
  { id: 'Alef', stack: "'Alef', 'Heebo', system-ui, sans-serif" },
  { id: 'Varela Round', stack: "'Varela Round', 'Heebo', system-ui, sans-serif" },
  { id: 'Frank Ruhl Libre', stack: "'Frank Ruhl Libre', Georgia, serif" },
  { id: 'Inter', stack: "'Inter', 'Heebo', system-ui, sans-serif" },
  { id: 'Arial', stack: "Arial, 'Heebo', Helvetica, sans-serif" },
  { id: 'Courier New', stack: "'Courier New', monospace" },
];

export const SUBTITLE_TEXT_COLORS = [
  '#FCD34D', // Amber (default)
  '#FFFFFF', // Pure white
  '#FACC15', // Cinema yellow
  '#67E8F9', // Cyan
  '#86EFAC', // Soft green
  '#F9A8D4', // Soft pink
] as const;

export const SUBTITLE_BG_COLORS = [
  '#000000', // Black
  '#18181B', // Dark zinc
  '#1E1B4B', // Deep indigo
  '#0F172A', // Slate navy
] as const;

export const DEFAULT_SUBTITLE_STYLE: SubtitleStyleSettings = {
  fontFamily: 'Heebo',
  fontSizePx: 18,
  textColor: '#FCD34D',
  backgroundColor: '#000000',
  backgroundOpacity: 75,
};

const HEX_COLOR_RE = /^#[0-9A-Fa-f]{6}$/;
const FONT_IDS = new Set<string>(SUBTITLE_FONTS.map((f) => f.id));

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  const num = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(max, Math.max(min, Math.round(num)));
}

export function sanitizeSubtitleStyle(raw?: Partial<SubtitleStyleSettings> | null): SubtitleStyleSettings {
  const base = DEFAULT_SUBTITLE_STYLE;
  if (!raw || typeof raw !== 'object') return { ...base };
  const fontFamily =
    typeof raw.fontFamily === 'string' && FONT_IDS.has(raw.fontFamily)
      ? (raw.fontFamily as SubtitleFontFamily)
      : base.fontFamily;
  const textColor =
    typeof raw.textColor === 'string' && HEX_COLOR_RE.test(raw.textColor.trim())
      ? raw.textColor.trim().toUpperCase()
      : base.textColor;
  const backgroundColor =
    typeof raw.backgroundColor === 'string' && HEX_COLOR_RE.test(raw.backgroundColor.trim())
      ? raw.backgroundColor.trim().toUpperCase()
      : base.backgroundColor;
  return {
    fontFamily,
    fontSizePx: clampInt(raw.fontSizePx, MIN_SUBTITLE_FONT_SIZE, MAX_SUBTITLE_FONT_SIZE, base.fontSizePx),
    textColor,
    backgroundColor,
    backgroundOpacity: clampInt(raw.backgroundOpacity, 0, 100, base.backgroundOpacity),
  };
}

export function subtitleFontStack(fontFamily: SubtitleFontFamily): string {
  return SUBTITLE_FONTS.find((f) => f.id === fontFamily)?.stack || SUBTITLE_FONTS[0].stack;
}

export function hexToRgba(hex: string, opacityPercent: number): string {
  const clean = HEX_COLOR_RE.test(hex.trim()) ? hex.trim().slice(1) : '000000';
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  const alpha = Math.max(0, Math.min(1, opacityPercent / 100));
  return `rgba(${r}, ${g}, ${b}, ${Number(alpha.toFixed(2))})`;
}

export function fontSizePxToEmbedTier(fontSizePx: number): EmbedFontSize {
  if (fontSizePx < 16) return 'small';
  if (fontSizePx >= 22) return 'large';
  return 'medium';
}

function load(): { language?: Language; theme?: Theme; subtitleStyle?: Partial<SubtitleStyleSettings> } {
  try {
    return JSON.parse(getItemWithLegacy(STORAGE_KEY) || '{}');
  } catch {
    return {};
  }
}

function save(language: Language, theme: Theme, subtitleStyle: SubtitleStyleSettings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ language, theme, subtitleStyle }));
  } catch {
    /* storage unavailable — preferences just won't persist */
  }
}

/** Reflect language/theme on <html> so CSS (data-theme) and layout direction follow. */
export function applyToDocument(language: Language, theme: Theme) {
  const root = document.documentElement;
  root.lang = language;
  root.dir = language === 'he' ? 'rtl' : 'ltr';
  root.dataset.theme = theme;
}

const saved = load();
const browserLanguage: Language =
  typeof navigator !== 'undefined' && navigator.language?.toLowerCase().startsWith('he') ? 'he' : 'en';

interface UiState {
  language: Language;
  theme: Theme;
  view: View;
  subtitleStyle: SubtitleStyleSettings;
  setLanguage: (language: Language) => void;
  setTheme: (theme: Theme) => void;
  setView: (view: View) => void;
  setSubtitleStyle: (patch: Partial<SubtitleStyleSettings>) => void;
  resetSubtitleStyle: () => void;
  /** The keyboard shortcuts window ("?") */
  shortcutsOpen: boolean;
  setShortcutsOpen: (open: boolean) => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  language: saved.language || browserLanguage,
  theme: saved.theme || 'dark',
  view: 'studio',
  subtitleStyle: sanitizeSubtitleStyle(saved.subtitleStyle),
  setLanguage: (language) => {
    save(language, get().theme, get().subtitleStyle);
    applyToDocument(language, get().theme);
    set({ language });
  },
  setTheme: (theme) => {
    save(get().language, theme, get().subtitleStyle);
    applyToDocument(get().language, theme);
    set({ theme });
  },
  setView: (view) => set({ view }),
  setSubtitleStyle: (patch) => {
    const next = sanitizeSubtitleStyle({ ...get().subtitleStyle, ...patch });
    save(get().language, get().theme, next);
    set({ subtitleStyle: next });
  },
  resetSubtitleStyle: () => {
    const next = { ...DEFAULT_SUBTITLE_STYLE };
    save(get().language, get().theme, next);
    set({ subtitleStyle: next });
  },
  shortcutsOpen: false,
  setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
}));
