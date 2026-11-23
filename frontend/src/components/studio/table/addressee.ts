import type { AddresseeInfo, Gender } from '../../../types';

/** At or above this confidence the translator was told the listener as a fact (see backend addressee.py) */
export const ADDRESSEE_SURE = 0.75;

/** The listener as male / female / group, or '' when it is not known. */
export const addresseeChoice = (a?: AddresseeInfo | null): Gender | '' => {
  if (!a) return '';
  if (a.number === 'plural') return 'group';
  if (a.gender === 'male' || a.gender === 'female') return a.gender;
  return '';
};

/** True when the line needs a listener (it has a "you" or an imperative) the app could not settle. */
export const isUncertainAddressee = (a?: AddresseeInfo | null): boolean =>
  !!a && !a.locked && a.confidence < ADDRESSEE_SURE;
