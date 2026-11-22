import React, { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle2, Cpu, Download, Loader2, XCircle } from 'lucide-react';
import { api } from '../../services/api';
import { useT } from '../../i18n';
import { useJobStore } from '../../store/useJobStore';
import type { GpuLibsStatus } from '../../types';
import { OPEN_GPU_LIBS_EVENT } from '../../utils/gpuLibs';

/** Set by "Don't ask again": the dialog no longer opens by itself (Settings can still open it). */
export const GPU_LIBS_DISMISSED_KEY = 'subbuild.gpuLibsDismissed';
/** After the update check's dialog had its chance, so the two never open together. */
export const GPU_OFFER_DELAY_MS = 4000;
export const GPU_POLL_MS = 3000;

function dismissedForever(): boolean {
  try {
    return localStorage.getItem(GPU_LIBS_DISMISSED_KEY) === '1';
  } catch {
    return false;
  }
}

type Phase = 'offer' | 'installing' | 'done' | 'failed';

/**
 * The Linux AppImage ships without NVIDIA's CUDA libraries (~1.5 GB). When the server finds an
 * NVIDIA card without them (hardware.gpu_libs_missing), this offers the one-time download
 * (POST /api/system/gpu-libs/install) and follows it until transcription can use the GPU.
 */
export const GpuLibsDialog: React.FC = () => {
  const t = useT();
  const hw = useJobStore((s) => s.systemInfo?.hardware);
  const fetchSystemInfo = useJobStore((s) => s.fetchSystemInfo);
  const [open, setOpen] = useState(false);
  const [phase, setPhase] = useState<Phase>('offer');
  const [status, setStatus] = useState<GpuLibsStatus | null>(null);
  const offered = useRef(false);
  const primaryRef = useRef<HTMLButtonElement | null>(null);

  const missing = !!hw?.gpu_libs_missing;

  // Offer it once per launch, a moment after startup
  useEffect(() => {
    if (!missing || offered.current || dismissedForever()) return;
    const timer = setTimeout(() => {
      offered.current = true;
      api
        .getGpuLibsStatus()
        .then((res) => {
          if (!res.supported || res.installed) return;
          setStatus(res);
          setPhase(res.installing ? 'installing' : 'offer');
          setOpen(true);
        })
        .catch(() => {
          /* server busy or older: Settings can still open it */
        });
    }, GPU_OFFER_DELAY_MS);
    return () => clearTimeout(timer);
  }, [missing]);

  // Settings' button
  useEffect(() => {
    const onOpen = () => {
      api
        .getGpuLibsStatus()
        .then((res) => {
          setStatus(res);
          setPhase(res.installed ? 'done' : res.installing ? 'installing' : 'offer');
          setOpen(true);
        })
        .catch(() => undefined);
    };
    window.addEventListener(OPEN_GPU_LIBS_EVENT, onOpen);
    return () => window.removeEventListener(OPEN_GPU_LIBS_EVENT, onOpen);
  }, []);

  // Follow the download (also while the dialog is closed: the result updates the hardware info)
  useEffect(() => {
    if (phase !== 'installing') return;
    const timer = setInterval(() => {
      api
        .getGpuLibsStatus()
        .then((res) => {
          setStatus(res);
          if (res.installing) return;
          if (res.installed) {
            setPhase('done');
            void fetchSystemInfo();
          } else {
            setPhase('failed');
          }
        })
        .catch(() => undefined);
    }, GPU_POLL_MS);
    return () => clearInterval(timer);
  }, [phase, fetchSystemInfo]);

  useEffect(() => {
    if (!open) return;
    primaryRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, phase]);

  const install = useCallback(() => {
    setPhase('installing');
    api
      .installGpuLibs()
      .then((res) => {
        setStatus(res);
        if (!res.installing && res.installed) {
          setPhase('done');
          void fetchSystemInfo();
        } else if (!res.installing && res.error) {
          setPhase('failed');
        }
      })
      .catch((e: unknown) => {
        setStatus((s) => ({ ...(s as GpuLibsStatus), error: e instanceof Error ? e.message : String(e) }));
        setPhase('failed');
      });
  }, [fetchSystemInfo]);

  if (!open) return null;

  const close = () => setOpen(false);
  const never = () => {
    try {
      localStorage.setItem(GPU_LIBS_DISMISSED_KEY, '1');
    } catch {
      /* storage unavailable - it is offered again next launch */
    }
    close();
  };
  const sizeGb = ((status?.download_size_mb || 1500) / 1000).toFixed(1);
  const gpu = hw?.gpu_name || 'NVIDIA GPU';
  const btn = 'px-3 py-1.5 rounded-lg text-xs cursor-pointer';

  return (
    <div className="fixed inset-0 z-[54] flex items-center justify-center bg-black/60 p-4" onClick={close}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="gpu-libs-title"
        data-testid="gpu-libs-dialog"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg bg-surface border border-line rounded-xl shadow-2xl p-5 space-y-4"
      >
        <h2 id="gpu-libs-title" className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-emerald-400 shrink-0" />
          {t('gpuLibs.title')}
        </h2>

        {phase === 'offer' && (
          <div className="space-y-2 text-xs text-zinc-300 leading-relaxed">
            <p data-testid="gpu-libs-body">{t('gpuLibs.body', { gpu, size: sizeGb })}</p>
            <p className="text-zinc-400">{t('gpuLibs.note')}</p>
          </div>
        )}
        {phase === 'installing' && (
          <p className="flex items-start gap-2 text-xs text-zinc-300 leading-relaxed" data-testid="gpu-libs-installing">
            <Loader2 className="w-4 h-4 animate-spin text-indigo-400 shrink-0 mt-0.5" />
            {t('gpuLibs.installing')}
          </p>
        )}
        {phase === 'done' && (
          <p className="flex items-start gap-2 text-xs text-emerald-300 leading-relaxed" data-testid="gpu-libs-done">
            <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
            {t('gpuLibs.done')}
          </p>
        )}
        {phase === 'failed' && (
          <div className="space-y-1.5" data-testid="gpu-libs-failed">
            <p className="flex items-start gap-2 text-xs text-rose-300">
              <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
              {t('gpuLibs.failed')}
            </p>
            {status?.error && (
              <pre className="text-[11px] text-zinc-400 whitespace-pre-wrap max-h-28 overflow-y-auto bg-sunken border border-line rounded-lg p-2" dir="ltr">
                {status.error}
              </pre>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
          <div className="flex flex-wrap gap-2">
            {phase === 'offer' && (
              <button type="button" onClick={never} data-testid="gpu-libs-never" className={`${btn} text-zinc-400 hover:text-zinc-200`}>
                {t('gpuLibs.never')}
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {(phase === 'offer' || phase === 'failed') && (
              <button
                type="button"
                onClick={close}
                data-testid="gpu-libs-later"
                className={`${btn} border border-line text-zinc-300 hover:border-zinc-500`}
              >
                {t('gpuLibs.later')}
              </button>
            )}
            {(phase === 'offer' || phase === 'failed') && (
              <button
                ref={primaryRef}
                type="button"
                onClick={install}
                data-testid="gpu-libs-install"
                className={`${btn} inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 font-semibold text-white`}
              >
                <Download className="w-3.5 h-3.5" />
                {phase === 'failed' ? t('gpuLibs.retry') : t('gpuLibs.install', { size: sizeGb })}
              </button>
            )}
            {(phase === 'installing' || phase === 'done') && (
              <button
                ref={primaryRef}
                type="button"
                onClick={close}
                data-testid="gpu-libs-close"
                className={`${btn} border border-line text-zinc-300 hover:border-zinc-500`}
              >
                {t('gpuLibs.close')}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
