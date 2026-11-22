/** Settings' "Enable GPU acceleration" button opens the GPU libraries dialog with this event. */
export const OPEN_GPU_LIBS_EVENT = 'subbuild:open-gpu-libs';

export function openGpuLibsDialog() {
  window.dispatchEvent(new Event(OPEN_GPU_LIBS_EVENT));
}
