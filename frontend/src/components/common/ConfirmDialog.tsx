import React, { useEffect, useRef } from 'react';
import { AlertTriangle } from 'lucide-react';
import { useDialogStore } from '../../store/useDialogStore';
import { useT } from '../../i18n';

export const ConfirmDialog: React.FC = () => {
  const request = useDialogStore((s) => s.request);
  const settle = useDialogStore((s) => s.settle);
  const t = useT();
  const okRef = useRef<HTMLButtonElement | null>(null);
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!request) return;
    (request.danger ? cancelRef : okRef).current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        settle(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [request, settle]);

  if (!request) return null;
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4" onClick={() => settle(false)}>
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby={request.message ? 'confirm-dialog-message' : undefined}
        data-testid="confirm-dialog"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md bg-surface border border-line rounded-xl shadow-2xl p-5 space-y-3"
      >
        <h2 id="confirm-dialog-title" className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
          {request.danger && <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />}
          {request.title}
        </h2>
        {request.message && (
          <p id="confirm-dialog-message" className="text-xs text-zinc-400 whitespace-pre-line leading-relaxed">
            {request.message}
          </p>
        )}
        <div className="flex justify-end gap-2 pt-1">
          <button
            ref={cancelRef}
            type="button"
            data-testid="confirm-dialog-cancel"
            onClick={() => settle(false)}
            className="px-3 py-1.5 rounded-lg border border-line text-xs text-zinc-300 hover:border-zinc-500 cursor-pointer"
          >
            {request.cancelLabel || t('ux.cancel')}
          </button>
          <button
            ref={okRef}
            type="button"
            data-testid="confirm-dialog-ok"
            onClick={() => settle(true)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold text-white cursor-pointer ${
              request.danger ? 'bg-rose-600 hover:bg-rose-500' : 'bg-indigo-600 hover:bg-indigo-500'
            }`}
          >
            {request.confirmLabel || t('ux.confirm')}
          </button>
        </div>
      </div>
    </div>
  );
};
