import React, { useEffect, useRef, useState } from 'react';
import { Download, ExternalLink, Sparkles } from 'lucide-react';
import { api } from '../../services/api';
import { useT } from '../../i18n';
import type { MessageKey } from '../../i18n';
import type { UpdateStatus } from '../../types';

/** The version the user chose to skip: no dialog for it, but a newer one is shown again. */
export const SKIPPED_VERSION_KEY = 'subbuild.updateSkippedVersion';
/** Wait a moment after launch so the check never competes with the app loading. */
export const STARTUP_CHECK_DELAY_MS = 2500;
/** Used when the release build hasn't attached the installer yet; the site always links the latest. */
export const WEBSITE_URL = 'https://subbuildstudio.com';

function readSkipped(): string | null {
  try {
    return localStorage.getItem(SKIPPED_VERSION_KEY);
  } catch {
    return null;
  }
}

function writeSkipped(version: string) {
  try {
    localStorage.setItem(SKIPPED_VERSION_KEY, version);
  } catch {
    /* storage unavailable - the dialog just shows again next launch */
  }
}

const STEPS: Record<'windows' | 'macos', MessageKey[]> = {
  windows: ['settings.newVersion.windowsStep1', 'settings.newVersion.windowsStep2', 'settings.newVersion.windowsStep3'],
  macos: ['settings.newVersion.macStep1', 'settings.newVersion.macStep2', 'settings.newVersion.macStep3'],
};

/**
 * A copy from the Windows installer or the macOS app can't patch itself, so at startup it checks
 * for a newer release and offers the new download with short instructions. Copies run from a Git
 * clone (run.bat / run.sh) update themselves before launch and never see this dialog.
 */
export const UpdateAvailableDialog: React.FC = () => {
  const t = useT();
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const downloadRef = useRef<HTMLAnchorElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      api
        .getUpdateStatus()
        .then((res) => {
          if (cancelled || !res.update_available || !res.installer) return;
          if (readSkipped() === res.latest_version) return;
          setStatus(res);
        })
        .catch(() => {
          /* offline or GitHub unreachable: say nothing, Settings can check again */
        });
    }, STARTUP_CHECK_DELAY_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (!status) return;
    downloadRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setStatus(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [status]);

  if (!status || !status.installer) return null;

  const close = () => setStatus(null);
  const skip = () => {
    writeSkipped(status.latest_version);
    close();
  };
  const downloadUrl = status.download_url || WEBSITE_URL;

  return (
    <div className="fixed inset-0 z-[55] flex items-center justify-center bg-black/60 p-4" onClick={close}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="update-dialog-title"
        data-testid="update-available-dialog"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg bg-surface border border-line rounded-xl shadow-2xl p-5 space-y-4"
      >
        <div className="space-y-1">
          <h2 id="update-dialog-title" className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400 shrink-0" />
            {t('settings.newVersion.title')}
          </h2>
          <p className="text-xs text-zinc-400 font-mono" dir="ltr" data-testid="update-dialog-versions">
            {t('settings.newVersion.versions', { current: status.current_version, latest: status.latest_version })}
          </p>
        </div>

        {status.release_notes && (
          <div className="space-y-1">
            <p className="text-[11px] font-semibold text-zinc-300">{t('settings.newVersion.whatsNew')}</p>
            <p
              className="text-xs text-zinc-400 whitespace-pre-line leading-relaxed max-h-32 overflow-y-auto bg-sunken border border-line rounded-lg p-2.5"
              dir="auto"
              data-testid="update-dialog-notes"
            >
              {status.release_notes}
            </p>
          </div>
        )}

        <div className="space-y-1.5">
          <p className="text-[11px] font-semibold text-zinc-300">{t('settings.newVersion.stepsTitle')}</p>
          <ol className="list-decimal ps-5 space-y-1 text-xs text-zinc-300 leading-relaxed" data-testid="update-dialog-steps">
            {STEPS[status.installer].map((key) => (
              <li key={key}>{t(key)}</li>
            ))}
          </ol>
          <p className="text-[11px] text-emerald-300/90">{t('settings.newVersion.keptNote')}</p>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={skip}
              data-testid="update-dialog-skip"
              className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 cursor-pointer"
            >
              {t('settings.newVersion.skip')}
            </button>
            <button
              type="button"
              onClick={close}
              data-testid="update-dialog-later"
              className="px-3 py-1.5 rounded-lg border border-line text-xs text-zinc-300 hover:border-zinc-500 cursor-pointer"
            >
              {t('settings.newVersion.later')}
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {status.release_url && (
              <a
                href={status.release_url}
                target="_blank"
                rel="noopener noreferrer"
                data-testid="update-dialog-release-page"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-line text-xs text-zinc-300 hover:border-zinc-500"
              >
                <ExternalLink className="w-3.5 h-3.5" />
                {t('settings.newVersion.releasePage')}
              </a>
            )}
            <a
              ref={downloadRef}
              href={downloadUrl}
              target="_blank"
              rel="noopener noreferrer"
              onClick={close}
              data-testid="update-dialog-download"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white"
            >
              <Download className="w-3.5 h-3.5" />
              {t('settings.newVersion.download')}
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};
