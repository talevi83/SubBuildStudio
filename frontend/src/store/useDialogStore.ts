import { create } from 'zustand';

export interface ConfirmOptions {
  title: string;
  message?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  /** A destructive action: red button, and the safe choice (Cancel) has the focus */
  danger?: boolean;
}

interface DialogState {
  request: (ConfirmOptions & { resolve: (ok: boolean) => void }) | null;
  confirm: (options: ConfirmOptions) => Promise<boolean>;
  settle: (ok: boolean) => void;
}

/** One themed, RTL-aware confirmation dialog for the whole app (instead of window.confirm). */
export const useDialogStore = create<DialogState>((set, get) => ({
  request: null,
  confirm: (options) =>
    new Promise<boolean>((resolve) => {
      get().request?.resolve(false); // a dialog that is replaced counts as cancelled
      set({ request: { ...options, resolve } });
    }),
  settle: (ok) => {
    const request = get().request;
    set({ request: null });
    request?.resolve(ok);
  },
}));

export const confirmDialog = (options: ConfirmOptions): Promise<boolean> => useDialogStore.getState().confirm(options);
