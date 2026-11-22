import React, { useEffect, useRef } from 'react';
import { RotateCcw, X } from 'lucide-react';
import { useT } from '../../i18n';
import {
  MAX_SUBTITLE_FONT_SIZE,
  MIN_SUBTITLE_FONT_SIZE,
  SUBTITLE_BG_COLORS,
  SUBTITLE_FONTS,
  SUBTITLE_TEXT_COLORS,
  useUiStore,
} from '../../store/useUiStore';
import type { SubtitleFontFamily } from '../../types';

interface Props {
  open: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
}

const OPACITY_PRESETS: Array<{ value: number; key: 'bgNone' | 'bgLight' | 'bgStandard' | 'bgSolid' }> = [
  { value: 0, key: 'bgNone' },
  { value: 50, key: 'bgLight' },
  { value: 75, key: 'bgStandard' },
  { value: 100, key: 'bgSolid' },
];

export const SubtitleStylePopover: React.FC<Props> = ({ open, onClose, anchorRef }) => {
  const t = useT();
  const subtitleStyle = useUiStore((s) => s.subtitleStyle);
  const setSubtitleStyle = useUiStore((s) => s.setSubtitleStyle);
  const resetSubtitleStyle = useUiStore((s) => s.resetSubtitleStyle);
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (!target) return;
      if (panelRef.current?.contains(target) || anchorRef.current?.contains(target)) return;
      onClose();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      }
    };
    document.addEventListener('mousedown', onPointerDown);
    window.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      window.removeEventListener('keydown', onKeyDown, true);
    };
  }, [open, onClose, anchorRef]);

  if (!open) return null;

  return (
    <div
      ref={panelRef}
      role="dialog"
      aria-label={t('player.style.title')}
      data-testid="subtitle-style-popover"
      onClick={(e) => e.stopPropagation()}
      className="absolute end-0 top-full mt-2 w-72 bg-surface border border-line rounded-xl shadow-2xl p-3.5 space-y-3.5 z-40"
    >
      <div className="flex items-center justify-between border-b border-line pb-2">
        <span className="text-xs font-semibold text-zinc-100">{t('player.style.title')}</span>
        <button
          type="button"
          onClick={onClose}
          className="text-zinc-400 hover:text-zinc-100 p-0.5 rounded cursor-pointer"
          aria-label={t('embed.close')}
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Font family */}
      <div className="space-y-1">
        <label htmlFor="subtitle-font-select" className="block text-[11px] font-medium text-zinc-400">
          {t('player.style.fontFamily')}
        </label>
        <select
          id="subtitle-font-select"
          data-testid="subtitle-font-family"
          value={subtitleStyle.fontFamily}
          onChange={(e) => setSubtitleStyle({ fontFamily: e.target.value as SubtitleFontFamily })}
          className="w-full bg-sunken border border-line rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-indigo-500 cursor-pointer"
        >
          {SUBTITLE_FONTS.map((font) => (
            <option key={font.id} value={font.id} style={{ fontFamily: font.stack }}>
              {font.id}
            </option>
          ))}
        </select>
      </div>

      {/* Bounded Font size */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[11px]">
          <span className="font-medium text-zinc-400">
            {t('player.style.fontSize', { min: MIN_SUBTITLE_FONT_SIZE, max: MAX_SUBTITLE_FONT_SIZE })}
          </span>
          <span
            data-testid="subtitle-size-value"
            className="font-mono text-indigo-300 font-semibold"
            dir="ltr"
          >
            {subtitleStyle.fontSizePx}px
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            data-testid="subtitle-size-dec"
            aria-label={t('player.style.smaller')}
            disabled={subtitleStyle.fontSizePx <= MIN_SUBTITLE_FONT_SIZE}
            onClick={() => setSubtitleStyle({ fontSizePx: subtitleStyle.fontSizePx - 1 })}
            className="w-6 h-6 flex items-center justify-center rounded-md bg-sunken border border-line text-xs font-bold text-zinc-200 hover:border-zinc-500 disabled:opacity-40 cursor-pointer disabled:cursor-not-allowed"
          >
            −
          </button>
          <input
            type="range"
            min={MIN_SUBTITLE_FONT_SIZE}
            max={MAX_SUBTITLE_FONT_SIZE}
            step={1}
            value={subtitleStyle.fontSizePx}
            onChange={(e) => setSubtitleStyle({ fontSizePx: Number(e.target.value) })}
            aria-label={t('player.style.fontSize', { min: MIN_SUBTITLE_FONT_SIZE, max: MAX_SUBTITLE_FONT_SIZE })}
            data-testid="subtitle-font-size"
            className="flex-1 accent-indigo-500 cursor-pointer"
          />
          <button
            type="button"
            data-testid="subtitle-size-inc"
            aria-label={t('player.style.larger')}
            disabled={subtitleStyle.fontSizePx >= MAX_SUBTITLE_FONT_SIZE}
            onClick={() => setSubtitleStyle({ fontSizePx: subtitleStyle.fontSizePx + 1 })}
            className="w-6 h-6 flex items-center justify-center rounded-md bg-sunken border border-line text-xs font-bold text-zinc-200 hover:border-zinc-500 disabled:opacity-40 cursor-pointer disabled:cursor-not-allowed"
          >
            +
          </button>
        </div>
      </div>

      {/* Text color */}
      <div className="space-y-1.5">
        <span className="block text-[11px] font-medium text-zinc-400">{t('player.style.textColor')}</span>
        <div className="flex items-center justify-between gap-1.5">
          <div className="flex items-center gap-1.5">
            {SUBTITLE_TEXT_COLORS.map((hex) => {
              const active = subtitleStyle.textColor.toUpperCase() === hex;
              return (
                <button
                  key={hex}
                  type="button"
                  onClick={() => setSubtitleStyle({ textColor: hex })}
                  title={hex}
                  aria-label={`${t('player.style.textColor')} ${hex}`}
                  data-testid={`subtitle-text-swatch-${hex.slice(1)}`}
                  className={`w-5 h-5 rounded-full border transition-transform cursor-pointer ${
                    active ? 'border-indigo-400 ring-2 ring-indigo-500/60 scale-110' : 'border-zinc-600 hover:scale-105'
                  }`}
                  style={{ backgroundColor: hex }}
                />
              );
            })}
          </div>
          <label className="flex items-center gap-1 text-[10px] text-zinc-400 cursor-pointer">
            <input
              type="color"
              value={subtitleStyle.textColor}
              onChange={(e) => setSubtitleStyle({ textColor: e.target.value })}
              aria-label={t('player.style.customColor')}
              data-testid="subtitle-text-color"
              className="w-6 h-6 rounded border border-line bg-transparent cursor-pointer p-0"
            />
          </label>
        </div>
      </div>

      {/* Background opacity & color */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-[11px]">
          <span className="font-medium text-zinc-400">{t('player.style.bgOpacity')}</span>
          <span
            data-testid="subtitle-bg-opacity-value"
            className="font-mono text-indigo-300 font-semibold"
            dir="ltr"
          >
            {subtitleStyle.backgroundOpacity}%
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={subtitleStyle.backgroundOpacity}
          onChange={(e) => setSubtitleStyle({ backgroundOpacity: Number(e.target.value) })}
          aria-label={t('player.style.bgOpacity')}
          data-testid="subtitle-bg-opacity"
          className="w-full accent-indigo-500 cursor-pointer"
        />
        <div className="grid grid-cols-4 gap-1">
          {OPACITY_PRESETS.map((preset) => {
            const active = subtitleStyle.backgroundOpacity === preset.value;
            return (
              <button
                key={preset.value}
                type="button"
                onClick={() => setSubtitleStyle({ backgroundOpacity: preset.value })}
                data-testid={`subtitle-opacity-preset-${preset.value}`}
                className={`py-1 rounded-md text-[10px] font-medium border transition-colors cursor-pointer ${
                  active
                    ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200'
                    : 'bg-sunken border-line text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {t(`player.style.${preset.key}`)}
              </button>
            );
          })}
        </div>

        <div className="flex items-center justify-between gap-2 pt-0.5">
          <span className="text-[11px] font-medium text-zinc-400">{t('player.style.bgColor')}</span>
          <div className="flex items-center gap-1.5">
            {SUBTITLE_BG_COLORS.map((hex) => {
              const active = subtitleStyle.backgroundColor.toUpperCase() === hex;
              return (
                <button
                  key={hex}
                  type="button"
                  onClick={() => setSubtitleStyle({ backgroundColor: hex })}
                  title={hex}
                  aria-label={`${t('player.style.bgColor')} ${hex}`}
                  className={`w-4 h-4 rounded-full border transition-transform cursor-pointer ${
                    active ? 'border-indigo-400 ring-2 ring-indigo-500/60 scale-110' : 'border-zinc-600 hover:scale-105'
                  }`}
                  style={{ backgroundColor: hex }}
                />
              );
            })}
            <input
              type="color"
              value={subtitleStyle.backgroundColor}
              onChange={(e) => setSubtitleStyle({ backgroundColor: e.target.value })}
              aria-label={t('player.style.customBgColor')}
              data-testid="subtitle-bg-color"
              className="w-5 h-5 rounded border border-line bg-transparent cursor-pointer p-0"
            />
          </div>
        </div>
      </div>

      {/* Reset */}
      <div className="pt-1 border-t border-line flex justify-end">
        <button
          type="button"
          onClick={resetSubtitleStyle}
          data-testid="subtitle-style-reset"
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] text-zinc-400 hover:text-zinc-100 hover:bg-sunken transition-colors cursor-pointer"
        >
          <RotateCcw className="w-3 h-3" />
          {t('player.style.reset')}
        </button>
      </div>
    </div>
  );
};
