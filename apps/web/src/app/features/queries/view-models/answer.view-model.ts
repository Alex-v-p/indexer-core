import type { AnswerPresentation } from '../models/answer.models';
import type { QueryResponse } from '../models/query-contract.models';

export interface QueryAnswerViewModel {
  mode: 'presentation' | 'legacy';
  presentation: AnswerPresentation | null;
  legacyAnswer: string;
}

/**
 * Structured presentation is additive and takes display precedence when present.
 * The persisted legacy answer is preserved unchanged as the compatibility fallback.
 */
export function buildQueryAnswerViewModel(
  result: Pick<QueryResponse, 'answer' | 'answer_presentation'>,
): QueryAnswerViewModel {
  const presentation = result.answer_presentation ?? null;
  return {
    mode: presentation ? 'presentation' : 'legacy',
    presentation,
    legacyAnswer: result.answer ?? 'No answer was returned.',
  };
}
