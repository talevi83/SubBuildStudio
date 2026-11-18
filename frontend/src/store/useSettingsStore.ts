import { create } from 'zustand';
import { api } from '../services/api';
import { useJobStore } from './useJobStore';
import type { AppSettings, CatalogModel, ProviderKeyRecord, ProviderSummary } from '../types';

interface SettingsState {
  settings: AppSettings | null;
  catalog: CatalogModel[];
  providers: Record<string, ProviderSummary>;
  keys: ProviderKeyRecord[];
  pricingChecked: string;
  pricingSource: string;
  loaded: boolean;
  refreshing: boolean;
  error: string | null;
  load: () => Promise<void>;
  save: (settings: AppSettings) => Promise<boolean>;
  refreshProviders: () => Promise<Record<string, string>>;
  syncVaultState: (payload: {
    settings: AppSettings;
    catalog: CatalogModel[];
    providers: Record<string, ProviderSummary>;
    keys: ProviderKeyRecord[];
  }) => void;
}

/** Server-side AI settings, multi-provider keys, and live-discovered model catalog. */
export const useSettingsStore = create<SettingsState>((set) => ({
  settings: null,
  catalog: [],
  providers: {},
  keys: [],
  pricingChecked: '',
  pricingSource: '',
  loaded: false,
  refreshing: false,
  error: null,
  load: async () => {
    try {
      const res = await api.getSettings();
      set({
        settings: res.settings,
        catalog: res.catalog,
        providers: res.providers || {},
        keys: res.keys || [],
        pricingChecked: res.pricing_checked,
        pricingSource: res.pricing_source,
        loaded: true,
        error: null,
      });
      useJobStore.getState().applySettingsDefaults(res.settings, false);
    } catch (e: any) {
      set({ error: e.message, loaded: true });
    }
  },
  save: async (settings) => {
    try {
      const res = await api.saveSettings(settings);
      set({
        settings: res.settings,
        catalog: res.catalog,
        providers: res.providers || {},
        keys: res.keys || [],
        error: null,
      });
      useJobStore.getState().applySettingsDefaults(res.settings, false);
      return true;
    } catch (e: any) {
      set({ error: e.message });
      return false;
    }
  },
  refreshProviders: async () => {
    set({ refreshing: true });
    try {
      const res = await api.refreshProviderModels();
      set({
        settings: res.settings,
        catalog: res.catalog,
        providers: res.providers || {},
        keys: res.keys || [],
        loaded: true,
        refreshing: false,
        error: null,
      });
      useJobStore.getState().applySettingsDefaults(res.settings, false);
      useJobStore.getState().fetchSystemInfo();
      return res.errors || {};
    } catch (e: any) {
      set({ refreshing: false, error: e.message });
      return { refresh: e.message || 'Refresh failed' };
    }
  },
  syncVaultState: (payload) => {
    set({
      settings: payload.settings,
      catalog: payload.catalog,
      providers: payload.providers || {},
      keys: payload.keys || [],
      loaded: true,
      error: null,
    });
    useJobStore.getState().applySettingsDefaults(payload.settings, false);
    useJobStore.getState().fetchSystemInfo();
  },
}));
