import { configure } from '@testing-library/react';

// How long waitFor / findBy* keep retrying: longer on CI (see test.env in vite.config.ts)
configure({ asyncUtilTimeout: Number(import.meta.env.VITE_TEST_ASYNC_TIMEOUT) || 1000 });
