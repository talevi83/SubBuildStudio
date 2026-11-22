import type { WordTiming } from '../types';

const DIR_MARKS_RE = /[\u200E\u200F\u202A-\u202E\u2066-\u2069]/g;

/**
 * Pair display words with stored word timestamps when exact word count aligns,
 * otherwise distribute [startTime, endTime] proportionally by character length.
 */
export function resolveDisplayWords(
  text: string,
  startTime: number,
  endTime: number,
  storedWords?: WordTiming[] | null
): WordTiming[] {
  const clean = (text || '').replace(DIR_MARKS_RE, '').trim();
  if (!clean) return [];

  const tokens = clean.split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return [];

  const start = Math.max(0, Number.isFinite(startTime) ? startTime : 0);
  const end = Math.max(start + 0.05, Number.isFinite(endTime) ? endTime : start + 0.05);

  if (storedWords && storedWords.length === tokens.length) {
    let cursor = start;
    return tokens.map((word, idx) => {
      const raw = storedWords[idx];
      const ws = Math.max(cursor, Math.min(end - 0.02, Number(raw.s ?? cursor)));
      let we = Math.max(ws + 0.02, Math.min(end, Number(raw.e ?? ws + 0.05)));
      if (idx === tokens.length - 1) we = Math.max(we, end);
      cursor = we;
      return { w: word, s: Number(ws.toFixed(3)), e: Number(we.toFixed(3)) };
    });
  }

  const weights = tokens.map((tok) => Math.max(1, tok.replace(/[.,!?;:"'״׳()[\]{}—–\-…]/g, '').length || tok.length));
  const totalWeight = weights.reduce((acc, w) => acc + w, 0) || tokens.length;
  const span = end - start;

  let cursor = start;
  return tokens.map((word, idx) => {
    const we = idx === tokens.length - 1 ? end : cursor + span * (weights[idx] / totalWeight);
    const out: WordTiming = {
      w: word,
      s: Number(cursor.toFixed(3)),
      e: Number(Math.max(cursor + 0.02, we).toFixed(3)),
    };
    cursor = we;
    return out;
  });
}

/** Return the index of the currently active word at `currentTime`, or -1 if none. */
export function findActiveWordIndex(words: WordTiming[], currentTime: number): number {
  if (words.length === 0) return -1;
  for (let i = 0; i < words.length; i += 1) {
    const w = words[i];
    if (currentTime >= w.s && (currentTime < w.e || i === words.length - 1)) {
      return i;
    }
  }
  if (currentTime >= words[words.length - 1].e) {
    return words.length - 1;
  }
  return -1;
}
