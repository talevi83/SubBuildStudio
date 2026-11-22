import React, { useEffect, useState } from 'react';
import { flushPendingEditsNow, hasUnsavedEdits, useJobStore } from './store/useJobStore';
import { useUiStore, applyToDocument } from './store/useUiStore';
import { useSettingsStore } from './store/useSettingsStore';
import { useT } from './i18n';
import { localizeError } from './i18n/serverErrors';
import { ConfirmDialog } from './components/common/ConfirmDialog';
import { ShortcutsDialog } from './components/common/ShortcutsDialog';
import { UpdateAvailableDialog } from './components/common/UpdateAvailableDialog';
import { GpuLibsDialog } from './components/common/GpuLibsDialog';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { MediaDropzone } from './components/ingest/MediaDropzone';
import { StreamSelector } from './components/ingest/StreamSelector';
import { PipelineConfig } from './components/ingest/PipelineConfig';
import { ModeSwitch } from './components/ingest/ModeSwitch';
import { SubtitleFilePicker } from './components/ingest/SubtitleFilePicker';
import { LivePipelineBar } from './components/monitor/LivePipelineBar';
import { VideoSyncPlayer } from './components/studio/VideoSyncPlayer';
import { SubtitleTable } from './components/studio/SubtitleTable';
import { ExportToolbar } from './components/studio/ExportToolbar';
import { SettingsScreen } from './components/settings/SettingsScreen';
import { HelpSection } from './components/settings/HelpSection';
import { FirstRunApiKeyModal } from './components/settings/ApiKeySection';

const MAX_ERRORS = 4;
const NOTICE_MS = 8000;

/** Errors stack up (newest last) until dismissed, so a second failure doesn't hide the first. */
export const ErrorToast: React.FC = () => {
  const error = useJobStore((s) => s.error);
  const clearError = useJobStore((s) => s.clearError);
  const t = useT();
  const [errors, setErrors] = useState<string[]>([]);

  useEffect(() => {
    if (!error) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setErrors((list) => (list.at(-1) === error ? list : [...list, error].slice(-MAX_ERRORS)));
  }, [error]);

  if (errors.length === 0) return null;
  const dismiss = (i: number) => {
    setErrors((list) => list.filter((_, j) => j !== i));
    // The same message failing again later must show up again
    if (errors.length === 1 || errors[i] === useJobStore.getState().error) clearError();
  };
  const dismissAll = () => {
    setErrors([]);
    clearError();
  };
  return (
    <div className="fixed bottom-4 end-4 z-50 max-w-md flex flex-col gap-2" data-testid="error-toasts">
      {errors.map((message, i) => (
        <div
          key={`${i}:${message}`}
          role="alert"
          aria-live="assertive"
          className="relative p-3 pe-8 rounded-lg bg-rose-950/90 border border-rose-500/40 text-rose-200 text-xs shadow-xl whitespace-pre-line"
        >
          {localizeError(message, t)}
          <button
            onClick={() => dismiss(i)}
            className="absolute top-2 end-2 text-rose-300 hover:text-rose-200 cursor-pointer"
            aria-label={t('app.dismiss')}
          >
            ✕
          </button>
        </div>
      ))}
      {errors.length > 1 && (
        <button
          onClick={dismissAll}
          className="self-end text-[11px] text-rose-300 hover:text-rose-100 underline cursor-pointer"
        >
          {t('ux.dismissAll')}
        </button>
      )}
    </div>
  );
};

/** "Line #4 deleted · Undo" - shown for a few seconds after an action that the history can undo. */
export const NoticeToast: React.FC = () => {
  const notice = useJobStore((s) => s.notice);
  const clearNotice = useJobStore((s) => s.clearNotice);
  const undo = useJobStore((s) => s.undo);
  const t = useT();

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => {
      if (useJobStore.getState().notice?.id === notice.id) clearNotice();
    }, NOTICE_MS);
    return () => clearTimeout(timer);
  }, [notice, clearNotice]);

  if (!notice) return null;
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="notice-toast"
      className="fixed bottom-4 start-1/2 -translate-x-1/2 rtl:translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2 rounded-lg bg-zinc-900/95 border border-line text-xs text-zinc-200 shadow-xl"
    >
      <span>{t(notice.key, notice.vars)}</span>
      {notice.undo && (
        <button
          data-testid="notice-undo"
          onClick={() => {
            clearNotice();
            void undo();
          }}
          className="font-semibold text-indigo-300 hover:text-indigo-200 cursor-pointer"
        >
          {t('ux.undo')}
        </button>
      )}
    </div>
  );
};

export const App: React.FC = () => {
  const activeJob = useJobStore((s) => s.activeJob);
  const view = useUiStore((s) => s.view);
  const syncing = useJobStore((s) => s.ingestMode === 'sync');
  const language = useUiStore((s) => s.language);
  const theme = useUiStore((s) => s.theme);
  const loadSettings = useSettingsStore((s) => s.load);
  const t = useT();

  useEffect(() => {
    applyToDocument(language, theme);
  }, [language, theme]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings, view]);

  // Closing the window (or reloading) right after typing: save the pending edit now, and ask the
  // browser to confirm while a save is still on its way (the request is sent with keepalive).
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      const unsaved = hasUnsavedEdits();
      flushPendingEditsNow();
      if (unsaved) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    const onHide = () => {
      if (document.visibilityState === 'hidden') flushPendingEditsNow();
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    document.addEventListener('visibilitychange', onHide);
    return () => {
      window.removeEventListener('beforeunload', onBeforeUnload);
      document.removeEventListener('visibilitychange', onHide);
    };
  }, []);

  // The shell is exactly one viewport tall and manages its own scrolling. If anything (a focus
  // jump, scrollIntoView...) still scrolls the document itself, snap it back instead of leaving
  // the layout shifted.
  useEffect(() => {
    const snapBack = () => {
      const el = document.scrollingElement;
      if (el && (el.scrollTop !== 0 || el.scrollLeft !== 0)) el.scrollTo(0, 0);
    };
    window.addEventListener('scroll', snapBack, { passive: true });
    return () => window.removeEventListener('scroll', snapBack);
  }, []);

  return (
    <div className="flex h-screen w-screen bg-app text-zinc-100 overflow-hidden font-sans">
      <ErrorToast />
      <NoticeToast />
      <ConfirmDialog />
      <ShortcutsDialog />
      <FirstRunApiKeyModal />
      <UpdateAvailableDialog />
      <GpuLibsDialog />
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        <Header />

        {view === 'settings' ? (
          <div className="flex-1 overflow-y-auto p-8">
            <SettingsScreen />
          </div>
        ) : view === 'help' ? (
          <div className="flex-1 overflow-y-auto p-8">
            <HelpSection />
          </div>
        ) : activeJob ? (
          <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
            <LivePipelineBar />
            <div className="flex-1 p-4 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0 overflow-hidden">
              <div className="lg:col-span-5 h-full min-h-0">
                <VideoSyncPlayer key={activeJob.id} />
              </div>
              <div className="lg:col-span-7 h-full min-h-0">
                <SubtitleTable />
              </div>
            </div>
            <ExportToolbar />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto p-8">
            <div className="max-w-4xl mx-auto space-y-6">
              <div className="text-center space-y-1 py-4">
                <h1 className="text-2xl font-bold tracking-tight text-zinc-50">{t('app.ingestTitle')}</h1>
                <p className="text-xs text-zinc-400">{t('app.ingestSubtitle')}</p>
              </div>
              <ModeSwitch />
              <MediaDropzone />
              {syncing && <SubtitleFilePicker />}
              <StreamSelector />
              <PipelineConfig />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
