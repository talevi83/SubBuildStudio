import { create } from 'zustand';
import { getItemWithLegacy } from '../utils/legacyStorage';
import type { EmbedFontSize, SubtitleFontFamily, SubtitleHighlightMode, SubtitleStyleSettings } from '../types';

export type Language = 'en' | 'he';
export type Theme = 'dark' | 'light';
export type View = 'studio' | 'settings' | 'help';

const STORAGE_KEY = 'subbuild.ui';

export const MIN_SUBTITLE_FONT_SIZE = 12;
export const MAX_SUBTITLE_FONT_SIZE = 36;
export const MIN_OUTLINE_WIDTH = 0;
export const MAX_OUTLINE_WIDTH = 6;
export const MIN_MARGIN_V = 8;
export const MAX_MARGIN_V = 120;

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
  outlineColor: '#000000',
  outlineWidth: 1.6,
  backgroundColor: '#000000',
  backgroundOpacity: 75,
  marginV: 16,
  bold: false,
  highlightMode: 'none',
  highlightColor: '#22D3EE',
};

export type SubtitleStylePresetId = 'cinema' | 'youtube' | 'reels' | 'podcast' | 'box';

export interface SubtitleStylePreset {
  id: SubtitleStylePresetId;
  style: SubtitleStyleSettings;
}

export const SUBTITLE_STYLE_PRESETS: SubtitleStylePreset[] = [
  {
    id: 'cinema',
    style: {
      fontFamily: 'Heebo',
      fontSizePx: 18,
      textColor: '#FFFFFF',
      outlineColor: '#000000',
      outlineWidth: 1.8,
      backgroundColor: '#000000',
      backgroundOpacity: 0,
      marginV: 18,
      bold: false,
      highlightMode: 'none',
      highlightColor: '#22D3EE',
    },
  },
  {
    id: 'youtube',
    style: {
      fontFamily: 'Assistant',
      fontSizePx: 20,
      textColor: '#FCD34D',
      outlineColor: '#000000',
      outlineWidth: 1.6,
      backgroundColor: '#000000',
      backgroundOpacity: 75,
      marginV: 16,
      bold: false,
      highlightMode: 'none',
      highlightColor: '#22D3EE',
    },
  },
  {
    id: 'reels',
    style: {
      fontFamily: 'Rubik',
      fontSizePx: 26,
      textColor: '#FFFFFF',
      outlineColor: '#000000',
      outlineWidth: 3.0,
      backgroundColor: '#000000',
      backgroundOpacity: 0,
      marginV: 48,
      bold: true,
      highlightMode: 'word',
      highlightColor: '#FACC15',
    },
  },
  {
    id: 'podcast',
    style: {
      fontFamily: 'Heebo',
      fontSizePx: 20,
      textColor: '#FFFFFF',
      outlineColor: '#000000',
      outlineWidth: 1.6,
      backgroundColor: '#18181B',
      backgroundOpacity: 80,
      marginV: 20,
      bold: false,
      highlightMode: 'none',
      highlightColor: '#38BDF8',
    },
  },
  {
    id: 'box',
    style: {
      fontFamily: 'Varela Round',
      fontSizePx: 19,
      textColor: '#FCD34D',
      outlineColor: '#000000',
      outlineWidth: 1.6,
      backgroundColor: '#000000',
      backgroundOpacity: 95,
      marginV: 16,
      bold: false,
      highlightMode: 'none',
      highlightColor: '#22D3EE',
    },
  },
];

const HEX_COLOR_RE = /^#[0-9A-Fa-f]{6}$/;
const FONT_IDS = new Set<string>(SUBTITLE_FONTS.map((f) => f.id));
const HIGHLIGHT_MODES = new Set<string>(['none', 'word', 'karaoke']);

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  const num = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(max, Math.max(min, Math.round(num)));
}

function clampDecimal(value: unknown, min: number, max: number, fallback: number): number {
  const num = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.round(Math.min(max, Math.max(min, num)) * 10) / 10;
}

function normalizeHex(value: unknown, fallback: string): string {
  return typeof value === 'string' && HEX_COLOR_RE.test(value.trim()) ? value.trim().toUpperCase() : fallback;
}

export function sanitizeSubtitleStyle(raw?: Partial<SubtitleStyleSettings> | null): SubtitleStyleSettings {
  const base = DEFAULT_SUBTITLE_STYLE;
  if (!raw || typeof raw !== 'object') return { ...base };
  const fontFamily =
    typeof raw.fontFamily === 'string' && FONT_IDS.has(raw.fontFamily)
      ? (raw.fontFamily as SubtitleFontFamily)
      : base.fontFamily;
  const highlightMode =
    typeof raw.highlightMode === 'string' && HIGHLIGHT_MODES.has(raw.highlightMode)
      ? (raw.highlightMode as SubtitleHighlightMode)
      : base.highlightMode;
  return {
    fontFamily,
    fontSizePx: clampInt(raw.fontSizePx, MIN_SUBTITLE_FONT_SIZE, MAX_SUBTITLE_FONT_SIZE, base.fontSizePx),
    textColor: normalizeHex(raw.textColor, base.textColor),
    outlineColor: normalizeHex(raw.outlineColor, base.outlineColor),
    outlineWidth: clampDecimal(raw.outlineWidth, MIN_OUTLINE_WIDTH, MAX_OUTLINE_WIDTH, base.outlineWidth),
    backgroundColor: normalizeHex(raw.backgroundColor, base.backgroundColor),
    backgroundOpacity: clampInt(raw.backgroundOpacity, 0, 100, base.backgroundOpacity),
    marginV: clampInt(raw.marginV, MIN_MARGIN_V, MAX_MARGIN_V, base.marginV),
    bold: typeof raw.bold === 'boolean' ? raw.bold : base.bold,
    highlightMode,
    highlightColor: normalizeHex(raw.highlightColor, base.highlightColor),
  };
}

export function buildSubtitleTextShadow(style: SubtitleStyleSettings): string {
  if (style.backgroundOpacity < 30 && style.outlineWidth > 0) {
    const c = hexToRgba(style.outlineColor, 95);
    const w = Math.max(1, Math.round(style.outlineWidth));
    return `0 1px ${w + 1}px ${c}, 0 0 ${w * 3}px ${c}, ${w}px 0 2px ${c}, -${w}px 0 2px ${c}`;
  }
  return '0 1px 2px rgba(0,0,0,0.6)';
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
