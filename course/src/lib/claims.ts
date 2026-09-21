// Sorting claims by how well supported they are: the three honesty levels used across the course.
export type ClaimLevel = 'established' | 'model' | 'research'

export const LEVEL_LABEL: Record<ClaimLevel, string> = {
  established: 'Established',
  model: 'Mental model',
  research: 'Active research',
}

export interface Claim { text: string; level: ClaimLevel; why: string }

/** Which answers are right. Unanswered claims (null) count as wrong. */
export const markClaims = (chosen: (ClaimLevel | null)[], claims: Claim[]): boolean[] =>
  claims.map((c, i) => chosen[i] === c.level)

export const scoreClaims = (chosen: (ClaimLevel | null)[], claims: Claim[]): number =>
  markClaims(chosen, claims).filter(Boolean).length
