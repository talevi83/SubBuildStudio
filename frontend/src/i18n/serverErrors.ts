import type { MessageKey } from './index';

type Translate = (key: MessageKey, vars?: Record<string, string | number>) => string;

/**
 * The server (and a few store helpers) report errors in English. These are the ones users actually
 * meet; they are shown in the UI's language. Anything else is shown as it came.
 */
const RULES: [RegExp, (m: RegExpMatchArray, t: Translate) => string][] = [
  [/^This window is not signed in/, (_m, t) => t('ux.err.notSignedIn')],
  [/^(The job|This job|The project) is still running/, (_m, t) => t('ux.err.stillRunning')],
  [/^Only neighbouring lines can be merged/, (_m, t) => t('table.mergeNotNeighbours')],
  [/^Select at least 2 lines to merge$/, (_m, t) => t('ux.err.selectTwo')],
  [/^Overlaps line #(\d+) \((.+)\)$/, (m, t) => t('ux.err.overlaps', { index: m[1], range: m[2] })],
  [/^Line is too short to split$/, (_m, t) => t('ux.err.tooShortToSplit')],
  [/^Line text cannot be split into two parts$/, (_m, t) => t('ux.err.cannotSplit')],
  [/^No project is open$/, (_m, t) => t('ux.err.noProject')],
  [/^Network paths/, (_m, t) => t('ux.err.networkPath')],
  [/^Unsupported media file format: '(.+)'$/, (m, t) => t('ux.err.unsupportedFormat', { ext: m[1] })],
  [/^(?:File not found on local disk|Target video file not found): (.+)$/, (m, t) => t('ux.err.fileNotFound', { path: m[1] })],
  [/^(?:Not enough free disk space|The disk filled up)/, (_m, t) => t('ux.err.diskSpace')],
  [/^The source video is no longer available$/, (_m, t) => t('ux.err.videoMissing')],
  [/^(?:Failed to fetch|NetworkError when attempting to fetch|Load failed)/, (_m, t) => t('ux.err.serverDown')],
  [/^Upload cancelled$/, (_m, t) => t('dropzone.uploadCancelled')],
];

export function localizeError(message: string, t: Translate): string {
  const saveFailed = message.match(/^Could not save subtitle: (.*)$/s);
  if (saveFailed) return t('ux.err.saveFailed', { message: localizeError(saveFailed[1], t) });
  for (const [pattern, render] of RULES) {
    const match = message.match(pattern);
    if (match) return render(match, t);
  }
  return message;
}
