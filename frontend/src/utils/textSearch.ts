import type { SubtitleSegment } from '../types';

/** How the table's search (filter, match navigation) and "Replace all" read the search text. */
export interface SearchOptions {
  useRegex: boolean;
  /** "Tom" does not match "tom" */
  matchCase: boolean;
  /** "cat" does not match inside "category"; works for Hebrew/Arabic letters too (not only \b) */
  wholeWord: boolean;
}

export const DEFAULT_SEARCH_OPTIONS: SearchOptions = { useRegex: false, matchCase: false, wholeWord: false };

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * One global RegExp for the search text, or the error of an invalid regular expression.
 * A regex written as /pattern/flags keeps its own flags (g is always added).
 */
export function buildSearchPattern(
  term: string,
  options: SearchOptions
): { regex: RegExp | null; error: string | null } {
  if (!term) return { regex: null, error: null };
  let source: string;
  let flags: string;
  const literal = options.useRegex ? term.match(/^\/(.*)\/([a-z]*)$/s) : null;
  if (literal) {
    source = literal[1];
    flags = literal[2].replace(/g/g, '');
  } else {
    source = options.useRegex ? term : escapeRegExp(term);
    flags = options.matchCase ? '' : 'i';
  }
  if (options.wholeWord) {
    // \b only knows ASCII letters; these look-arounds also treat Hebrew letters as word characters
    source = `(?<![\\p{L}\\p{N}_])(?:${source})(?![\\p{L}\\p{N}_])`;
    if (!flags.includes('u') && !flags.includes('v')) flags += 'u';
  }
  try {
    return { regex: new RegExp(source, `${flags}g`), error: null };
  } catch (err) {
    return { regex: null, error: err instanceof Error ? err.message : String(err) };
  }
}

/** Whether `text` contains a match (safe to call repeatedly with the same global RegExp). */
export function hasMatch(regex: RegExp, text: string | null | undefined): boolean {
  if (!text) return false;
  regex.lastIndex = 0;
  return regex.test(text);
}

export interface ReplacePatch {
  id: number;
  before: string;
  after: string;
}

/** The lines "Replace all" would change, with their text before and after. Nothing is saved. */
export function planReplacements(
  segments: SubtitleSegment[],
  field: 'original_text' | 'translated_text',
  regex: RegExp,
  replacement: string,
  useRegex: boolean
): ReplacePatch[] {
  const patches: ReplacePatch[] = [];
  for (const s of segments) {
    const before = s[field] || '';
    if (!s.id || !hasMatch(regex, before)) continue;
    regex.lastIndex = 0;
    // Plain text: "$1" or "$&" in the replacement is taken literally
    const after = useRegex ? before.replace(regex, replacement) : before.replace(regex, () => replacement);
    if (after !== before) patches.push({ id: s.id, before, after });
  }
  return patches;
}
