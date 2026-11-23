import type { AddresseeInfo, Gender } from '../../../types';

/** At or above this confidence the translator was told the listener as a fact (see backend addressee.py) */
export const ADDRESSEE_SURE = 0.75;

/** The listener as one of the choices the user can pick, or '' when it is not known. */
export const addresseeChoice = (a?: AddresseeInfo | null): Gender | '' => {
  if (!a) return '';
  if (a.number === 'plural') return 'group';
  if (a.gender === 'male' || a.gender === 'female') return a.gender;
  return '';
};

/** True when the line needs a listener (it has a "you" or an imperative) the app could not settle. */
export const isUncertainAddressee = (a?: AddresseeInfo | null): boolean =>
  !!a && !a.locked && a.confidence < ADDRESSEE_SURE;

/** What the server stores for a listener picked by hand: shown right away while the save is on its way. */
export const manualAddressee = (choice: Gender | null | undefined): AddresseeInfo | null => {
  if (!choice) return null;
  return {
    gender: choice === 'group' ? 'mixed' : choice,
    number: choice === 'group' ? 'plural' : 'singular',
    confidence: 1,
    source: 'manual',
    locked: true,
  };
};
