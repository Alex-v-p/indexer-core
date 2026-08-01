import type { AnswerPresentation } from '../models/answer.models';
import type { QueryResponse } from '../models/query-contract.models';

export interface ProductOutcomeViewModel {
  outcome: 'clarification_required' | 'no_evidence';
  title: string;
  body: string;
  candidates: string[];
}

export interface QueryAnswerViewModel {
  mode: 'outcome' | 'presentation' | 'legacy';
  presentation: AnswerPresentation | null;
  legacyAnswer: string;
  outcome: ProductOutcomeViewModel | null;
}

/**
 * Structured presentation is additive and takes display precedence when present.
 * The persisted legacy answer is preserved unchanged as the compatibility fallback.
 */
export function buildQueryAnswerViewModel(
  result: Pick<QueryResponse, 'answer' | 'answer_presentation'> &
    Partial<Pick<QueryResponse, 'product_outcome' | 'subject_scope'>>,
): QueryAnswerViewModel {
  const presentation = result.answer_presentation ?? null;
  const productOutcome = result.product_outcome ?? result.subject_scope?.product_outcome ?? null;
  const outcome = buildProductOutcome(productOutcome, result.subject_scope ?? null);
  return {
    mode: outcome ? 'outcome' : presentation ? 'presentation' : 'legacy',
    presentation,
    legacyAnswer: result.answer ?? 'No answer was returned.',
    outcome,
  };
}

function buildProductOutcome(
  outcome: QueryResponse['product_outcome'],
  scope: QueryResponse['subject_scope'],
): ProductOutcomeViewModel | null {
  if (outcome === 'clarification_required') {
    return {
      outcome,
      title: 'Clarification required',
      body: clarificationMessage(scope?.clarification_reason ?? null),
      candidates: (scope?.ambiguity_candidates ?? []).map((candidate) => candidate.name),
    };
  }
  if (outcome === 'no_evidence') {
    return {
      outcome,
      title: 'No evidence in the resolved scope',
      body: 'The request completed safely, but no eligible document evidence was available for the resolved subject scope.',
      candidates: [],
    };
  }
  return null;
}

function clarificationMessage(reason: string | null): string {
  const messages: Record<string, string> = {
    unknown_explicit_subject: 'One or more selected subjects are no longer available. Refresh the subject list and choose again.',
    unknown_named_project: 'A project named in the question could not be matched. Choose an explicit project or clarify its name.',
    ambiguous_project_name: 'The project name matches more than one project. Choose the intended project before running retrieval.',
    hard_scope_document_limit_exceeded: 'The resolved project scope contains too many documents to search safely. Narrow the selected subjects.',
    comparison_project_limit_exceeded: 'The comparison names more projects than can be searched in one run. Narrow the comparison.',
    multiple_projects_require_comparison_intent: 'Multiple projects were selected, but the question does not clearly request a comparison. Clarify the intended project or comparison.',
  };
  return reason ? messages[reason] ?? `The subject scope needs clarification (${reason.replaceAll('_', ' ')}).` : 'The subject scope needs clarification before retrieval can continue.';
}
