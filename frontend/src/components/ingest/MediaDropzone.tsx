import React, { useCallback, useRef, useState } from 'react';
import { api, UploadCancelled } from '../../services/api';
import { useJobStore } from '../../store/useJobStore';
import { useT } from '../../i18n';
import { localizeError } from '../../i18n/serverErrors';
import { formatClock } from '../../utils/time';
import { naturalSortBy, pickLocalMediaFiles, readClipboardText, resolveDroppedFilePath } from '../../utils/desktop';
import { PasteContextMenu } from '../common/PasteContextMenu';
import type { ProbeResponse } from '../../types';
import { Upload, HardDrive, AlertCircle, Loader2, CheckCircle2, Layers, Plus, X } from 'lucide-react';

export const MediaDropzone: React.FC = () => {
  const setProbeResult = useJobStore((s) => s.setProbeResult);
  const setBatchProbes = useJobStore((s) => s.setBatchProbes);
  const addBatchProbes = useJobStore((s) => s.addBatchProbes);
  const removeBatchProbe = useJobStore((s) => s.removeBatchProbe);
  const probeResult = useJobStore((s) => s.probeResult);
  const batchProbes = useJobStore((s) => s.batchProbes);
  const ingestMode = useJobStore((s) => s.ingestMode);
  const t = useT();
  const [localPath, setLocalPath] = useState('');
  const [isProbing, setIsProbing] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number; name: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  // Hashing a big file to find an earlier identical upload can take as long as uploading it
  const [hashProgress, setHashProgress] = useState<number | null>(null);
  const uploadAbort = useRef<AbortController | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pathInputRef = useRef<HTMLInputElement | null>(null);
  const [pasteMenu, setPasteMenu] = useState<{ x: number; y: number } | null>(null);
  const closePasteMenu = useCallback(() => setPasteMenu(null), []);

  const probePathDirectly = async (rawPath: string, append = false) => {
    const cleaned = rawPath.trim().replace(/^["']+|["']+$/g, '');
    if (!cleaned) return;
    setLocalPath(cleaned);
    setIsProbing(true);
    setError(null);
    try {
      const result = await api.probePath(cleaned);
      if (append) {
        addBatchProbes([result]);
        setLocalPath('');
      } else {
        setProbeResult(result);
        if (result.file_path && !result.is_upload) {
          setLocalPath(result.file_path);
        }
      }
    } catch (err: any) {
      setError(err.message || t('dropzone.probeFailed'));
    } finally {
      setIsProbing(false);
    }
  };

  const probePathsBatch = async (rawPaths: string[], append = false) => {
    const cleaned = Array.from(
      new Set(rawPaths.map((p) => p.trim().replace(/^["']+|["']+$/g, '')).filter(Boolean))
    );
    if (cleaned.length === 0) return;
    if (cleaned.length === 1) {
      await probePathDirectly(cleaned[0], append);
      return;
    }
    const sorted = naturalSortBy(cleaned, (p) => p.split(/[\\/]/).pop() || p);
    setIsProbing(true);
    setError(null);
    const results: ProbeResponse[] = [];
    try {
      for (let i = 0; i < sorted.length; i++) {
        const p = sorted[i];
        const baseName = p.split(/[\\/]/).pop() || p;
        setBatchProgress({ current: i + 1, total: sorted.length, name: baseName });
        const result = await api.probePath(p);
        results.push(result);
      }
      if (append) addBatchProbes(results);
      else setBatchProbes(results);
      setLocalPath('');
    } catch (err: any) {
      if (results.length > 0) {
        if (append) addBatchProbes(results);
        else setBatchProbes(results);
      }
      setError(err.message || t('dropzone.probeFailed'));
    } finally {
      setIsProbing(false);
      setBatchProgress(null);
    }
  };

  // Right-click on the path field offers only "Paste" (the text goes in at the cursor, like Ctrl+V)
  const pasteIntoPath = async () => {
    setPasteMenu(null);
    const text = (await readClipboardText())?.replace(/[\r\n]+/g, ' ').trim();
    if (text === undefined || text === null) {
      setError(t('dropzone.pasteFailed'));
      return;
    }
    if (!text) return;
    const input = pathInputRef.current;
    const start = input?.selectionStart ?? localPath.length;
    const end = input?.selectionEnd ?? localPath.length;
    const next = localPath.slice(0, start) + text + localPath.slice(end);
    setLocalPath(next);
    setError(null);
    requestAnimationFrame(() => {
      if (!input) return;
      input.focus();
      const caret = start + text.length;
      input.setSelectionRange(caret, caret);
    });
  };

  const handleProbeLocalPath = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!localPath.trim()) return;
    await probePathDirectly(localPath);
  };

  const handleFileUpload = async (file: File, append = false) => {
    setIsProbing(true);
    setError(null);
    const controller = new AbortController();
    uploadAbort.current = controller;
    try {
      // Same file uploaded before? Reuse it instead of copying gigabytes again.
      const existing = await api.findExistingUpload(file, { signal: controller.signal, onProgress: setHashProgress });
      setHashProgress(null);
      if (existing) {
        if (append) addBatchProbes([existing]);
        else setProbeResult(existing);
        return;
      }
      setUploadProgress(0);
      const uploaded = await api.uploadFile(file, setUploadProgress, controller.signal);
      if (append) addBatchProbes([uploaded]);
      else setProbeResult(uploaded);
    } catch (err: any) {
      setError(err instanceof UploadCancelled ? t('dropzone.uploadCancelled') : err.message || t('dropzone.uploadFailed'));
    } finally {
      uploadAbort.current = null;
      setIsProbing(false);
      setUploadProgress(null);
      setHashProgress(null);
    }
  };

  const handleBatchFileUploads = async (files: File[], append = false) => {
    const sorted = naturalSortBy(files, (f) => f.name);
    setIsProbing(true);
    setError(null);
    const controller = new AbortController();
    uploadAbort.current = controller;
    const results: ProbeResponse[] = [];
    try {
      for (let i = 0; i < sorted.length; i++) {
        const file = sorted[i];
        setBatchProgress({ current: i + 1, total: sorted.length, name: file.name });
        setHashProgress(null);
        setUploadProgress(null);
        const existing = await api.findExistingUpload(file, {
          signal: controller.signal,
          onProgress: setHashProgress,
        });
        setHashProgress(null);
        if (existing) {
          results.push(existing);
          continue;
        }
        setUploadProgress(0);
        results.push(await api.uploadFile(file, setUploadProgress, controller.signal));
      }
      if (append) addBatchProbes(results);
      else setBatchProbes(results);
    } catch (err: any) {
      if (results.length > 0) {
        if (append) addBatchProbes(results);
        else setBatchProbes(results);
      }
      setError(err instanceof UploadCancelled ? t('dropzone.uploadCancelled') : err.message || t('dropzone.uploadFailed'));
    } finally {
      uploadAbort.current = null;
      setIsProbing(false);
      setUploadProgress(null);
      setHashProgress(null);
      setBatchProgress(null);
    }
  };

  const onDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (isProbing) return;

    const droppedFiles = Array.from(e.dataTransfer.files || []);
    if (droppedFiles.length > 0) {
      // Must call resolveDroppedFilePath synchronously before any await so dataTransfer.files is still live
      const pathPromises = droppedFiles.map((f) => resolveDroppedFilePath(f, e.dataTransfer));
      const resolvedList = await Promise.all(pathPromises);
      const resolvedPaths = resolvedList.filter((p): p is string => Boolean(p));
      if (resolvedPaths.length === droppedFiles.length) {
        if (resolvedPaths.length === 1) {
          await probePathDirectly(resolvedPaths[0]);
        } else {
          await probePathsBatch(resolvedPaths);
        }
        return;
      }
      if (droppedFiles.length === 1) {
        await handleFileUpload(droppedFiles[0]);
      } else {
        await handleBatchFileUploads(droppedFiles);
      }
      return;
    }

    const droppedText = e.dataTransfer.getData('text/plain')?.trim();
    if (droppedText) {
      const lines = droppedText
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean);
      if (lines.length > 1) {
        await probePathsBatch(lines);
      } else {
        await probePathDirectly(droppedText);
      }
    }
  };

  const openFilePicker = async (append = false) => {
    if (isProbing) return;
    const pickedPaths = await pickLocalMediaFiles();
    if (pickedPaths !== null) {
      if (pickedPaths.length === 1) {
        await probePathDirectly(pickedPaths[0], append);
      } else if (pickedPaths.length > 1) {
        await probePathsBatch(pickedPaths, append);
      }
      return;
    }
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = 'video/*,audio/*,.mkv,.mp4,.avi,.mov';
    input.onchange = (e: any) => {
      const files: File[] = Array.from(e.target.files || []);
      if (files.length === 1) void handleFileUpload(files[0], append);
      else if (files.length > 1) void handleBatchFileUploads(files, append);
    };
    input.click();
  };

  const handleBrowseClick = () => openFilePicker(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!isDragging) setIsDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
          setIsDragging(false);
        }
      }}
      onDrop={(e) => void onDrop(e)}
      className="bg-surface border border-line rounded-xl p-5 shadow-lg space-y-4"
    >
      <div className="border-b border-line pb-3 flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-indigo-400" />
            {t('dropzone.title')}
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">{t('dropzone.subtitle')}</p>
        </div>
        {ingestMode !== 'sync' && (
          <span
            data-testid="batch-feature-badge"
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-500/15 border border-indigo-500/35 text-indigo-300 text-[11px] font-semibold"
          >
            <Layers className="w-3.5 h-3.5" />
            {t('dropzone.batchBadge')}
          </span>
        )}
      </div>

      <div
        onClick={() => void handleBrowseClick()}
        data-testid="media-dropzone-area"
        className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors flex flex-col items-center justify-center cursor-pointer ${
          isDragging
            ? 'border-indigo-500 bg-indigo-500/10'
            : 'border-line hover:border-indigo-500/50 bg-sunken/50'
        }`}
      >
        <Upload className="w-7 h-7 text-indigo-400 mb-2" />
        <p className="text-xs font-medium text-zinc-200">{t('dropzone.dropTitle')}</p>
        <p className="text-[11px] text-zinc-500 mt-1">{t('dropzone.dropHint')}</p>
        {ingestMode !== 'sync' && (
          <div
            data-testid="batch-callout"
            className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/25 text-[11px] text-indigo-200 max-w-xl"
          >
            <Layers className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" />
            <span>{t('dropzone.batchCallout')}</span>
          </div>
        )}
        {isProbing && (
          <div className="mt-3 w-full max-w-sm space-y-1.5">
            <div className="flex items-center justify-center gap-2 text-xs text-indigo-400 font-medium">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              {batchProgress
                ? t('dropzone.inspectingBatch', {
                    current: batchProgress.current,
                    total: batchProgress.total,
                    name: batchProgress.name,
                  })
                : hashProgress !== null
                ? t('dropzone.checkingDuplicate', { percent: Math.round(hashProgress * 100) })
                : uploadProgress !== null && uploadProgress < 1
                  ? t('dropzone.uploading', { percent: Math.round(uploadProgress * 100) })
                  : t('dropzone.inspectingMedia')}
            </div>
            {(uploadProgress !== null || hashProgress !== null) && (
              <div className="h-1.5 rounded-full bg-line overflow-hidden" dir="ltr">
                <div
                  className="h-full bg-indigo-500 transition-all"
                  style={{ width: `${(hashProgress ?? uploadProgress ?? 0) * 100}%` }}
                />
              </div>
            )}
            {(hashProgress !== null || (uploadProgress !== null && uploadProgress < 1)) && (
              <button
                type="button"
                data-testid="upload-cancel"
                onClick={(e) => {
                  e.stopPropagation(); // the drop area opens the file picker on click
                  uploadAbort.current?.abort();
                }}
                className="px-3 py-1 rounded-md border border-line text-[11px] text-zinc-300 hover:border-rose-500/60 hover:text-rose-300 cursor-pointer"
              >
                {t('dropzone.cancel')}
              </button>
            )}
          </div>
        )}
      </div>

      <form onSubmit={(e) => void handleProbeLocalPath(e)} className="space-y-2">
        <label className="block text-xs font-medium text-zinc-300">{t('dropzone.pathLabel')}</label>
        <div className="flex gap-2">
          <input
            ref={pathInputRef}
            type="text"
            dir="ltr"
            value={localPath}
            onChange={(e) => setLocalPath(e.target.value)}
            onContextMenu={(e) => {
              e.preventDefault();
              setPasteMenu({ x: e.clientX, y: e.clientY });
            }}
            data-testid="local-path-input"
            placeholder={t('dropzone.pathPlaceholder')}
            className="flex-1 bg-sunken border border-line rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-indigo-500 font-mono"
          />
          <button
            type="submit"
            disabled={isProbing || !localPath.trim()}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:text-zinc-500 text-white text-xs font-semibold rounded-lg transition-colors flex items-center gap-2 cursor-pointer"
          >
            {isProbing ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                {t('dropzone.inspecting')}
              </>
            ) : (
              t('dropzone.inspect')
            )}
          </button>
          {probeResult && ingestMode !== 'sync' && (
            <button
              type="button"
              disabled={isProbing || !localPath.trim()}
              onClick={() => void probePathDirectly(localPath, true)}
              data-testid="path-add-to-batch-btn"
              className="px-3 py-2 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 text-indigo-200 border border-indigo-500/35 text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer whitespace-nowrap"
            >
              <Plus className="w-3.5 h-3.5" />
              {t('dropzone.addToBatch')}
            </button>
          )}
        </div>
        <p className="text-[11px] text-zinc-500">{t('dropzone.pathHint')}</p>
        {pasteMenu && (
          <PasteContextMenu x={pasteMenu.x} y={pasteMenu.y} onPaste={() => void pasteIntoPath()} onClose={closePasteMenu} />
        )}
      </form>

      {error && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{localizeError(error, t)}</span>
        </div>
      )}

      {batchProbes.length > 1 ? (
        <div
          className="p-3.5 rounded-lg bg-emerald-500/10 border border-emerald-500/25 space-y-2.5"
          data-testid="batch-ingest-list"
        >
          <div className="flex items-center justify-between gap-2 text-xs text-emerald-300 flex-wrap">
            <span className="flex items-center gap-2 font-semibold">
              <Layers className="w-4 h-4 flex-shrink-0" />
              {t('dropzone.batchReadyTitle', {
                count: batchProbes.length,
                size: Math.round(batchProbes.reduce((acc, p) => acc + p.file_size, 0) / (1024 * 1024)),
                duration: formatClock(batchProbes.reduce((acc, p) => acc + p.duration, 0)),
              })}
            </span>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => void openFilePicker(true)}
                data-testid="batch-add-more-btn"
                className="inline-flex items-center gap-1 text-[11px] font-medium text-indigo-300 hover:text-indigo-200 cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                {t('dropzone.batchAddMore')}
              </button>
              <button
                type="button"
                onClick={() => setBatchProbes([])}
                data-testid="batch-clear-all"
                className="text-[11px] text-zinc-400 hover:text-zinc-200 underline cursor-pointer"
              >
                {t('dropzone.batchClear')}
              </button>
            </div>
          </div>
          <div className="max-h-48 overflow-y-auto space-y-1.5 pe-1">
            {batchProbes.map((item, idx) => (
              <div
                key={item.file_path}
                className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-md bg-sunken/80 border border-line text-xs"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 flex-shrink-0">
                    #{idx + 1}
                  </span>
                  <span className="text-zinc-200 font-medium truncate" dir="auto" title={item.file_path}>
                    {item.file_name}
                  </span>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0 text-[11px] text-zinc-400">
                  <span className="font-mono-time">{formatClock(item.duration)}</span>
                  <span>·</span>
                  <span>{Math.round(item.file_size / (1024 * 1024))} MB</span>
                  <button
                    type="button"
                    onClick={() => removeBatchProbe(item.file_path)}
                    data-testid={`batch-remove-${idx}`}
                    title={t('dropzone.batchRemove')}
                    aria-label={t('dropzone.batchRemove')}
                    className="p-0.5 rounded text-zinc-400 hover:text-rose-400 hover:bg-rose-500/10 cursor-pointer"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        probeResult && (
          <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-2 min-w-0">
              <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
              <span>
                {t('dropzone.probed', {
                  name: probeResult.file_name,
                  size: Math.round(probeResult.file_size / (1024 * 1024)),
                  duration: formatClock(probeResult.duration),
                })}
              </span>
            </div>
            {ingestMode !== 'sync' && (
              <button
                type="button"
                disabled={isProbing}
                onClick={() => void openFilePicker(true)}
                data-testid="batch-add-more-btn"
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-500/15 hover:bg-indigo-500/25 border border-indigo-500/35 text-indigo-200 text-[11px] font-semibold transition-colors cursor-pointer flex-shrink-0"
              >
                <Plus className="w-3.5 h-3.5" />
                {t('dropzone.batchAddMore')}
              </button>
            )}
          </div>
        )
      )}
    </div>
  );
};
