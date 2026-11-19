import type { SubtitleSegment } from '../../../types';

/** On-screen text has its own place on the screen; every other kind of line shares one lane. */
const lane = (s: SubtitleSegment) => (s.segment_type === 'ON_SCREEN' ? 'ON_SCREEN' : 'MAIN');

/**
 * Whether the checked lines can be merged: two or more lines of one lane that follow each other in
 * time order. Merging #1 with #4 would make one line covering #2 and #3 (the server refuses it too).
 */
export function areNeighbours(segments: SubtitleSegment[], checkedIds: ReadonlySet<number>): boolean {
  if (checkedIds.size < 2) return false;
  const picked = segments.filter((s) => s.id != null && checkedIds.has(s.id));
  if (picked.length !== checkedIds.size) return false;
  const lanes = new Set(picked.map(lane));
  if (lanes.size !== 1) return false;
  const [only] = lanes;
  const ordered = [...segments]
    .filter((s) => lane(s) === only)
    .sort((a, b) => a.start_time - b.start_time || a.index - b.index);
  const positions = ordered
    .map((s, i) => (s.id != null && checkedIds.has(s.id) ? i : -1))
    .filter((i) => i >= 0);
  return positions.every((p, i) => i === 0 || p === positions[i - 1] + 1);
}
