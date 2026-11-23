import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../../services/api';
import type { Glossary, GlossaryCharacter, SpeakerProfile, SubtitleSegment } from '../../types';
import { useT } from '../../i18n';
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Loader2,
  Play,
  Plus,
  Search,
  Sparkles,
  Trash2,
  Users,
  Volume2,
  X,
} from 'lucide-react';

interface Props {
  glossary: Glossary | null | undefined;
  profile: string | null | undefined;
  canTranslate: boolean;
  speakers?: Record<string, SpeakerProfile> | null;
  segments?: SubtitleSegment[];
  onSaveSpeakers?: (speakers: Record<string, SpeakerProfile>) => Promise<boolean>;
  onDiarize?: () => Promise<boolean>;
  onMatchSpeakers?: (opts?: { overwrite_manual?: boolean; auto_merge?: boolean }) => Promise<boolean>;
  onMergeSpeakers?: (sourceSpeakerIds: string[], targetSpeakerId: string | null) => Promise<boolean>;
  onPreviewTime?: (timeSeconds: number) => void;
  onClose: () => void;
  onSave: (glossary: Glossary, thenTranslate: boolean, profile: string) => Promise<void>;
}

type Term = { source: string; translation: string; notes?: string };
type TabId = 'speakers' | 'characters' | 'terms';

const inputClass =
  'w-full bg-sunken border border-line rounded px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500';

const fmtClock = (sec: number) => {
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

function pickSampleQuotes(segs: SubtitleSegment[]): SubtitleSegment[] {
  if (segs.length === 0) return [];
  const clear = segs.filter((s) => (s.original_text || '').trim().length >= 18);
  const pool = clear.length > 0 ? clear : [...segs].sort((a, b) => (b.original_text || '').length - (a.original_text || '').length);
  if (pool.length === 1) return [pool[0]];
  const mid = pool[Math.min(pool.length - 1, Math.max(1, Math.floor(pool.length / 2)))];
  return pool[0].index === mid.index ? [pool[0]] : [pool[0], mid];
}

/** Edit cast voices (with auto-matching, sample quotes, and merging), characters, and terms. */
export const GlossaryDialog: React.FC<Props> = ({
  glossary,
  profile: initialProfile,
  canTranslate,
  speakers: propSpeakers,
  segments,
  onSaveSpeakers,
  onDiarize,
  onMatchSpeakers,
  onMergeSpeakers,
  onPreviewTime,
  onClose,
  onSave,
}) => {
  const t = useT();
  const hasSpeakersOrDiarize = Object.keys(propSpeakers ?? {}).length > 0 || Boolean(onDiarize);
  const [activeTab, setActiveTab] = useState<TabId>(() => (hasSpeakersOrDiarize ? 'speakers' : 'characters'));
  const [editedSpeakers, setEditedSpeakers] = useState<Record<string, SpeakerProfile> | null>(null);
  const [diarizing, setDiarizing] = useState(false);
  const [matching, setMatching] = useState(false);
  const [merging, setMerging] = useState(false);
  const [speakerSearch, setSpeakerSearch] = useState('');
  const [minorOpen, setMinorOpen] = useState(false);
  const [prevGlossary, setPrevGlossary] = useState(glossary);
  const [characters, setCharacters] = useState<GlossaryCharacter[]>(() => glossary?.characters ?? []);
  const [terms, setTerms] = useState<Term[]>(() => glossary?.terms ?? []);
  const [register, setRegister] = useState(glossary?.register_notes ?? '');
  const [saving, setSaving] = useState(false);
  const [profile, setProfile] = useState(initialProfile ?? '');
  const [profiles, setProfiles] = useState<string[]>([]);

  if (glossary !== prevGlossary) {
    setPrevGlossary(glossary);
    setCharacters(glossary?.characters ?? []);
    setTerms(glossary?.terms ?? []);
    setRegister(glossary?.register_notes ?? '');
  }

  useEffect(() => {
    api.listGlossaryProfiles().then(setProfiles).catch(() => setProfiles([]));
  }, []);

  const currentSpeakers = useMemo(() => editedSpeakers ?? propSpeakers ?? {}, [editedSpeakers, propSpeakers]);
  const speakerEntries = useMemo(() => Object.entries(currentSpeakers), [currentSpeakers]);

  const { segmentsBySpeaker, totalSpoken } = useMemo(() => {
    const map: Record<string, SubtitleSegment[]> = {};
    let total = 0;
    for (const seg of segments || []) {
      if (seg.segment_type !== 'ON_SCREEN') total += 1;
      if (seg.speaker) {
        if (!map[seg.speaker]) map[seg.speaker] = [];
        map[seg.speaker].push(seg);
      }
    }
    return { segmentsBySpeaker: map, totalSpoken: Math.max(1, total) };
  }, [segments]);

  const { emptySpeakerIds, activeEntries, mainEntries, minorEntries } = useMemo(() => {
    const emptyIds: string[] = [];
    const active: Array<[string, SpeakerProfile]> = [];
    for (const entry of speakerEntries) {
      const count = segmentsBySpeaker[entry[0]]?.length ?? 0;
      if (count === 0) emptyIds.push(entry[0]);
      else active.push(entry);
    }
    active.sort((a, b) => (segmentsBySpeaker[b[0]]?.length ?? 0) - (segmentsBySpeaker[a[0]]?.length ?? 0));

    let main = active;
    let minor: Array<[string, SpeakerProfile]> = [];
    if (active.length > 6) {
      const atLeast3 = active.filter(([id]) => (segmentsBySpeaker[id]?.length ?? 0) >= 3);
      if (atLeast3.length >= 3) {
        main = atLeast3;
        minor = active.filter(([id]) => (segmentsBySpeaker[id]?.length ?? 0) < 3);
      } else {
        main = active.slice(0, 8);
        minor = active.slice(8);
      }
    }
    return { emptySpeakerIds: emptyIds, activeEntries: active, mainEntries: main, minorEntries: minor };
  }, [speakerEntries, segmentsBySpeaker]);

  const searchQuery = speakerSearch.trim().toLowerCase();
  const filterBySearch = (entries: Array<[string, SpeakerProfile]>) => {
    if (!searchQuery) return entries;
    return entries.filter(([id, sp]) => {
      if (id.toLowerCase().includes(searchQuery)) return true;
      if ((sp.label || '').toLowerCase().includes(searchQuery)) return true;
      if ((sp.character_name || '').toLowerCase().includes(searchQuery)) return true;
      const segs = segmentsBySpeaker[id] || [];
      return segs.some(
        (s) =>
          (s.original_text || '').toLowerCase().includes(searchQuery) ||
          (s.translated_text || '').toLowerCase().includes(searchQuery)
      );
    });
  };

  const visibleMain = filterBySearch(mainEntries);
  const visibleMinor = filterBySearch(minorEntries);
  const showMinorExpanded = minorOpen || Boolean(searchQuery);
  const showNoiseBanner = emptySpeakerIds.length > 0 || activeEntries.length > 20;

  const updateSpeaker = (id: string, patch: Partial<SpeakerProfile>) => {
    const base = editedSpeakers ?? propSpeakers ?? {};
    if (!base[id]) return;
    setEditedSpeakers({ ...base, [id]: { ...base[id], ...patch } });
  };

  const handleLinkCharacter = (speakerId: string, charIndexStr: string) => {
    const base = editedSpeakers ?? propSpeakers ?? {};
    const curSp = base[speakerId];
    if (!curSp) return;
    if (charIndexStr === '') {
      updateSpeaker(speakerId, { character_name: null, confidence: null });
      setCharacters((list) => list.map((c) => (c.speaker_id === speakerId ? { ...c, speaker_id: '' } : c)));
      return;
    }
    const idx = Number(charIndexStr);
    const ch = characters[idx];
    if (!ch) return;
    const nextLabel = (ch.target_name || ch.name || '').trim() || curSp.label;
    const nextGender = ch.gender === 'male' || ch.gender === 'female' ? ch.gender : curSp.gender;
    updateSpeaker(speakerId, {
      label: nextLabel,
      gender: nextGender,
      character_name: ch.name,
      confidence: 'high',
    });
    setCharacters((list) =>
      list.map((c, j) => {
        if (j === idx) return { ...c, speaker_id: speakerId };
        if (c.speaker_id === speakerId) return { ...c, speaker_id: '' };
        return c;
      })
    );
  };

  const runDetectSpeakers = async () => {
    if (!onDiarize || diarizing || matching || merging) return;
    setDiarizing(true);
    try {
      const ok = await onDiarize();
      if (ok) setEditedSpeakers(null);
    } finally {
      setDiarizing(false);
    }
  };

  const runMatchSpeakers = async () => {
    if (!onMatchSpeakers || matching || diarizing || merging) return;
    setMatching(true);
    try {
      if (editedSpeakers && onSaveSpeakers && Object.keys(editedSpeakers).length > 0) {
        await onSaveSpeakers(editedSpeakers);
      }
      const ok = await onMatchSpeakers({ overwrite_manual: false, auto_merge: true });
      if (ok) setEditedSpeakers(null);
    } finally {
      setMatching(false);
    }
  };

  const runMerge = async (sourceIds: string[], targetId: string | null) => {
    if (!onMergeSpeakers || sourceIds.length === 0 || merging || diarizing || matching) return;
    setMerging(true);
    try {
      const ok = await onMergeSpeakers(sourceIds, targetId);
      if (ok) setEditedSpeakers(null);
    } finally {
      setMerging(false);
    }
  };

  const setChar = (i: number, patch: Partial<GlossaryCharacter>) =>
    setCharacters((list) => list.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  const setTerm = (i: number, patch: Partial<Term>) =>
    setTerms((list) => list.map((c, j) => (j === i ? { ...c, ...patch } : c)));

  const submit = async (thenTranslate: boolean) => {
    setSaving(true);
    try {
      if (editedSpeakers && onSaveSpeakers && Object.keys(editedSpeakers).length > 0) {
        const spOk = await onSaveSpeakers(editedSpeakers);
        if (!spOk) return;
      }
      await onSave(
        { summary: glossary?.summary ?? '', characters, terms, register_notes: register },
        thenTranslate,
        profile
      );
    } finally {
      setSaving(false);
    }
  };

  const busy = diarizing || matching || merging || saving;

  const renderSpeakerCard = (id: string, sp: SpeakerProfile) => {
    const segs = segmentsBySpeaker[id] || [];
    const count = segs.length;
    const pct = Math.max(1, Math.round((count / totalSpoken) * 100));
    const quotes = pickSampleQuotes(segs);
    const linkedCharIdx = characters.findIndex(
      (c) =>
        (c.speaker_id && c.speaker_id === id) ||
        (sp.character_name && c.name.trim().toLowerCase() === sp.character_name.trim().toLowerCase())
    );
    const mergeTargets = activeEntries.filter(([otherId]) => otherId !== id);

    return (
      <div
        key={id}
        data-testid={`speaker-card-${id}`}
        className="p-2.5 rounded-lg bg-sunken/70 border border-line space-y-2"
      >
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="color"
            value={sp.color || '#38BDF8'}
            onChange={(e) => updateSpeaker(id, { color: e.target.value })}
            aria-label={`${id} color`}
            className="w-6 h-6 rounded border border-line bg-transparent cursor-pointer p-0 flex-shrink-0"
          />
          <input
            value={sp.label}
            onChange={(e) => updateSpeaker(id, { label: e.target.value })}
            placeholder={id}
            data-testid={`speaker-label-${id}`}
            className="flex-1 min-w-[8rem] bg-surface border border-line rounded px-2 py-1 text-xs text-zinc-100 focus:outline-none focus:border-indigo-500"
            dir="auto"
          />
          {characters.length > 0 && (
            <select
              value={linkedCharIdx >= 0 ? String(linkedCharIdx) : ''}
              onChange={(e) => handleLinkCharacter(id, e.target.value)}
              data-testid={`speaker-link-${id}`}
              aria-label={t('glossary.linkCharacter')}
              className="max-w-[11rem] bg-surface border border-line rounded px-2 py-1 text-xs text-zinc-300 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="">{linkedCharIdx >= 0 ? t('glossary.unlinkCharacter') : t('glossary.linkCharacter')}</option>
              {characters.map((ch, idx) => (
                <option key={`${ch.name}-${idx}`} value={String(idx)}>
                  {ch.target_name || ch.name}
                </option>
              ))}
            </select>
          )}
          <select
            value={sp.gender || 'unknown'}
            onChange={(e) => updateSpeaker(id, { gender: e.target.value as SpeakerProfile['gender'] })}
            data-testid={`speaker-gender-${id}`}
            aria-label={t('speakers.genderLabel')}
            className="w-24 bg-surface border border-line rounded px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
          >
            <option value="unknown">{t('speakers.unknown')}</option>
            <option value="male">{t('speakers.male')}</option>
            <option value="female">{t('speakers.female')}</option>
          </select>
          <span
            className="text-[10px] text-zinc-400 font-mono whitespace-nowrap px-1.5 py-0.5 rounded bg-surface border border-line"
            title={id}
          >
            {t('speakers.cues', { count })} · {pct}%
          </span>
          {onMergeSpeakers && (
            <select
              value=""
              disabled={busy}
              onChange={(e) => {
                const val = e.target.value;
                if (!val) return;
                void runMerge([id], val === '__UNASSIGN__' ? null : val);
              }}
              data-testid={`speaker-merge-${id}`}
              aria-label={t('glossary.mergeInto')}
              className="max-w-[9.5rem] bg-surface border border-line rounded px-1.5 py-1 text-[11px] text-zinc-300 focus:outline-none focus:border-indigo-500 cursor-pointer disabled:opacity-50"
            >
              <option value="">{t('glossary.mergeInto')}</option>
              {mergeTargets.map(([tid, tsp]) => (
                <option key={tid} value={tid}>
                  {t('glossary.mergeTarget', { label: tsp.label || tid })} ({segmentsBySpeaker[tid]?.length ?? 0})
                </option>
              ))}
              <option value="__UNASSIGN__">{t('glossary.unassignSpeaker')}</option>
            </select>
          )}
        </div>

        {(sp.confidence === 'high' || sp.confidence === 'medium' || quotes.length > 0) && (
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            {(sp.confidence === 'high' || sp.confidence === 'medium') && (
              <span
                title={sp.evidence || undefined}
                data-testid={`speaker-match-badge-${id}`}
                className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-medium ${
                  sp.confidence === 'high'
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                    : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                }`}
              >
                {sp.confidence === 'high' ? '✓' : '~'}{' '}
                {sp.confidence === 'high' ? t('glossary.matchedHigh') : t('glossary.matchedMedium')}
              </span>
            )}
            {quotes.map((q) => (
              <div
                key={q.index}
                className="flex items-center gap-1.5 max-w-full px-2 py-0.5 rounded bg-surface/80 border border-line/70 text-zinc-400"
              >
                <button
                  type="button"
                  data-testid={`speaker-preview-${id}`}
                  disabled={!onPreviewTime}
                  onClick={() => onPreviewTime?.(q.start_time)}
                  title={t('glossary.previewVoice')}
                  className="inline-flex items-center gap-1 text-[10px] font-mono text-indigo-300 hover:text-indigo-200 disabled:opacity-50 cursor-pointer flex-shrink-0"
                  dir="ltr"
                >
                  <Play className="w-2.5 h-2.5 fill-current" />
                  {fmtClock(q.start_time)}
                </button>
                <span className="truncate max-w-[16rem] text-zinc-300" dir="auto">
                  “{q.original_text.replace(/\s+/g, ' ').trim()}”
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className="w-full max-w-4xl max-h-[86vh] flex flex-col overflow-hidden bg-surface border border-line rounded-xl shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Sticky Header */}
        <div className="px-5 pt-4 pb-3 border-b border-line space-y-3 flex-shrink-0 bg-sunken/40">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <Users className="w-4 h-4 text-indigo-400" />
                {t('glossary.modalTitle')}
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">{t('glossary.help')}</p>
            </div>
            <button
              type="button"
              onClick={onClose}
              aria-label={t('glossary.close')}
              className="p-1 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Tab Bar */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {hasSpeakersOrDiarize && (
              <button
                type="button"
                data-testid="tab-speakers"
                onClick={() => setActiveTab('speakers')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                  activeTab === 'speakers'
                    ? 'bg-indigo-600/20 text-indigo-200 border border-indigo-500/40'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60 border border-transparent'
                }`}
              >
                {t('glossary.tabSpeakers')} ({activeEntries.length})
              </button>
            )}
            <button
              type="button"
              data-testid="tab-characters"
              onClick={() => setActiveTab('characters')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                activeTab === 'characters'
                  ? 'bg-indigo-600/20 text-indigo-200 border border-indigo-500/40'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60 border border-transparent'
              }`}
            >
              {t('glossary.tabCharacters')} ({characters.length})
            </button>
            <button
              type="button"
              data-testid="tab-terms"
              onClick={() => setActiveTab('terms')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                activeTab === 'terms'
                  ? 'bg-indigo-600/20 text-indigo-200 border border-indigo-500/40'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60 border border-transparent'
              }`}
            >
              {t('glossary.tabTerms')} ({terms.length})
            </button>
          </div>
        </div>

        {/* Scrollable Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {activeTab === 'speakers' && hasSpeakersOrDiarize && (
            <section className="space-y-3" data-testid="speakers-section">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  {onDiarize && (
                    <button
                      type="button"
                      onClick={() => void runDetectSpeakers()}
                      disabled={busy}
                      data-testid="diarize-audio-btn"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/40 text-xs font-medium text-indigo-200 disabled:opacity-50 cursor-pointer"
                    >
                      {diarizing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Volume2 className="w-3.5 h-3.5" />}
                      {diarizing ? t('glossary.detectingSpeakers') : t('glossary.detectSpeakers')}
                    </button>
                  )}
                  {onMatchSpeakers && activeEntries.length > 0 && (
                    <button
                      type="button"
                      onClick={() => void runMatchSpeakers()}
                      disabled={busy}
                      data-testid="match-speakers-btn"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/35 text-xs font-medium text-emerald-200 disabled:opacity-50 cursor-pointer"
                    >
                      {matching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                      {matching ? t('glossary.matchingSpeakers') : t('glossary.matchSpeakers')}
                    </button>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  {activeEntries.length > 0 && (
                    <span className="text-[11px] text-zinc-400">
                      {t('glossary.castSummary', { main: mainEntries.length, minor: minorEntries.length })}
                    </span>
                  )}
                  {activeEntries.length > 6 && (
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 text-zinc-500 absolute start-2 top-1/2 -translate-y-1/2 pointer-events-none" />
                      <input
                        value={speakerSearch}
                        onChange={(e) => setSpeakerSearch(e.target.value)}
                        placeholder={t('glossary.searchSpeakers')}
                        data-testid="speaker-search-input"
                        className="bg-sunken border border-line rounded-lg ps-7 pe-2 py-1 text-xs text-zinc-200 w-44 focus:outline-none focus:border-indigo-500"
                        dir="auto"
                      />
                    </div>
                  )}
                </div>
              </div>

              {showNoiseBanner && (
                <div
                  data-testid="legacy-speakers-banner"
                  className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs flex flex-wrap items-center justify-between gap-2"
                >
                  <div className="flex items-start gap-2 flex-1 min-w-[14rem]">
                    <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                    <span>
                      {t('glossary.legacyNoiseWarning', {
                        count: speakerEntries.length,
                        empty: emptySpeakerIds.length,
                      })}
                    </span>
                  </div>
                  {emptySpeakerIds.length > 0 && onMergeSpeakers && (
                    <button
                      type="button"
                      data-testid="clean-empty-speakers-btn"
                      disabled={busy}
                      onClick={() => void runMerge(emptySpeakerIds, null)}
                      className="px-2.5 py-1 rounded-md bg-amber-500/20 hover:bg-amber-500/30 border border-amber-400/40 text-[11px] font-medium text-amber-100 cursor-pointer disabled:opacity-50"
                    >
                      {t('glossary.cleanEmptySpeakers', { count: emptySpeakerIds.length })}
                    </button>
                  )}
                </div>
              )}

              {activeEntries.length === 0 ? (
                <p className="text-xs text-zinc-500 py-4 text-center">{t('glossary.noSpeakersYet')}</p>
              ) : (
                <div className="space-y-2">
                  {visibleMain.map(([id, sp]) => renderSpeakerCard(id, sp))}

                  {minorEntries.length > 0 && (
                    <div className="pt-1">
                      <button
                        type="button"
                        data-testid="minor-speakers-toggle"
                        onClick={() => setMinorOpen((o) => !o)}
                        className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-sunken/50 hover:bg-sunken border border-line text-xs text-zinc-300 font-medium cursor-pointer"
                      >
                        <span>{t('glossary.minorSpeakersToggle', { count: visibleMinor.length })}</span>
                        {showMinorExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                      {showMinorExpanded && (
                        <div className="mt-2 space-y-2">{visibleMinor.map(([id, sp]) => renderSpeakerCard(id, sp))}</div>
                      )}
                    </div>
                  )}

                  {visibleMain.length === 0 && visibleMinor.length === 0 && (
                    <p className="text-xs text-zinc-500 py-4 text-center">{t('glossary.noMatchingSpeakers')}</p>
                  )}
                </div>
              )}
            </section>
          )}

          {activeTab === 'characters' && (
            <section className="space-y-2.5" data-testid="characters-section">
              {characters.length === 0 && <p className="text-xs text-zinc-500">{t('glossary.empty')}</p>}
              {characters.map((c, i) => {
                const linkedSp = c.speaker_id ? currentSpeakers[c.speaker_id] : undefined;
                return (
                  <div key={i} className="grid grid-cols-[1fr_1fr_7rem_1.4fr_auto] gap-2 items-center">
                    <input
                      data-testid="glossary-char-name"
                      value={c.name}
                      onChange={(e) => setChar(i, { name: e.target.value })}
                      placeholder={t('glossary.name')}
                      className={inputClass}
                      dir="auto"
                    />
                    <input
                      data-testid="glossary-char-target"
                      value={c.target_name ?? ''}
                      onChange={(e) => setChar(i, { target_name: e.target.value })}
                      placeholder={t('glossary.targetName')}
                      className={inputClass}
                      dir="auto"
                    />
                    <select
                      data-testid="glossary-char-gender"
                      value={c.gender || 'unknown'}
                      onChange={(e) => setChar(i, { gender: e.target.value })}
                      className={inputClass}
                      aria-label={t('glossary.gender')}
                    >
                      <option value="unknown">{t('glossary.unknown')}</option>
                      <option value="male">{t('glossary.male')}</option>
                      <option value="female">{t('glossary.female')}</option>
                      <option value="mixed group">{t('glossary.group')}</option>
                    </select>
                    <div className="flex items-center gap-1.5 min-w-0">
                      <input
                        value={c.notes ?? ''}
                        onChange={(e) => setChar(i, { notes: e.target.value })}
                        placeholder={t('glossary.notes')}
                        className={inputClass}
                        dir="auto"
                      />
                      {linkedSp && (
                        <span
                          title={t('glossary.linkedVoice', { label: linkedSp.label })}
                          className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-indigo-500/15 border border-indigo-500/30 text-[10px] text-indigo-200 whitespace-nowrap flex-shrink-0"
                        >
                          <span
                            className="w-1.5 h-1.5 rounded-full"
                            style={{ backgroundColor: linkedSp.color || '#38BDF8' }}
                          />
                          {linkedSp.label}
                        </span>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => setCharacters((l) => l.filter((_, j) => j !== i))}
                      title={t('glossary.remove')}
                      className="p-1 text-zinc-500 hover:text-rose-400 cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })}
              <button
                type="button"
                data-testid="glossary-add-character"
                onClick={() =>
                  setCharacters((l) => [...l, { name: '', target_name: '', gender: 'unknown', notes: '' }])
                }
                className="flex items-center gap-1 text-xs text-indigo-300 hover:text-indigo-200 cursor-pointer"
              >
                <Plus className="w-3 h-3" />
                {t('glossary.add')}
              </button>
            </section>
          )}

          {activeTab === 'terms' && (
            <section className="space-y-4" data-testid="terms-section">
              <div className="space-y-2">
                <h3 className="text-[11px] uppercase tracking-wider text-zinc-500 font-semibold">
                  {t('glossary.terms')}
                </h3>
                {terms.map((term, i) => (
                  <div key={i} className="grid grid-cols-[1fr_1fr_auto] gap-2 items-center">
                    <input
                      value={term.source}
                      onChange={(e) => setTerm(i, { source: e.target.value })}
                      placeholder={t('glossary.source')}
                      className={inputClass}
                      dir="auto"
                    />
                    <input
                      value={term.translation}
                      onChange={(e) => setTerm(i, { translation: e.target.value })}
                      placeholder={t('glossary.translation')}
                      className={inputClass}
                      dir="auto"
                    />
                    <button
                      type="button"
                      onClick={() => setTerms((l) => l.filter((_, j) => j !== i))}
                      title={t('glossary.remove')}
                      className="p-1 text-zinc-500 hover:text-rose-400 cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  onClick={() => setTerms((l) => [...l, { source: '', translation: '' }])}
                  className="flex items-center gap-1 text-xs text-indigo-300 hover:text-indigo-200 cursor-pointer"
                >
                  <Plus className="w-3 h-3" />
                  {t('glossary.add')}
                </button>
              </div>

              <label className="block space-y-1 text-[11px] text-zinc-400">
                <span>{t('glossary.profile')}</span>
                <input
                  value={profile}
                  onChange={(e) => setProfile(e.target.value)}
                  list="glossary-profiles"
                  placeholder={t('glossary.profilePlaceholder')}
                  className={inputClass}
                  dir="auto"
                  data-testid="glossary-profile"
                />
                <datalist id="glossary-profiles">
                  {profiles.map((p) => (
                    <option key={p} value={p} />
                  ))}
                </datalist>
                <span className="block text-zinc-500">{t('glossary.profileHelp')}</span>
              </label>

              <label className="block space-y-1 text-[11px] text-zinc-400">
                <span>{t('glossary.registerNotes')}</span>
                <input
                  value={register}
                  onChange={(e) => setRegister(e.target.value)}
                  className={inputClass}
                  dir="auto"
                />
              </label>
            </section>
          )}
        </div>

        {/* Sticky Footer */}
        <div className="px-5 py-3 border-t border-line bg-sunken/40 flex justify-end gap-2 flex-shrink-0">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg text-xs text-zinc-300 border border-line hover:text-white cursor-pointer"
          >
            {t('glossary.close')}
          </button>
          <button
            type="button"
            onClick={() => void submit(false)}
            disabled={busy}
            className="px-3 py-1.5 rounded-lg text-xs text-zinc-200 border border-zinc-700 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 cursor-pointer"
            data-testid="glossary-save"
          >
            {t('glossary.save')}
          </button>
          <button
            type="button"
            onClick={() => void submit(true)}
            disabled={busy || !canTranslate}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white cursor-pointer"
          >
            {t('glossary.saveAndTranslate')}
          </button>
        </div>
      </div>
    </div>
  );
};
