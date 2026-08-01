import { AnswerPresentation } from '../models/answer.models';
import { buildQueryAnswerViewModel } from './answer.view-model';

describe('buildQueryAnswerViewModel', () => {
  it('prefers structured presentation without rewriting the legacy answer', () => {
    const presentation: AnswerPresentation = {
      schema_version: '1.0',
      outcome: 'partial',
      title: 'Partial answer',
      body: 'Supported information only [1].',
      supported_information: ['Supported information.'],
      unresolved_information: ['Missing information.'],
      citation_count: 1,
    };

    const result = buildQueryAnswerViewModel({
      answer: 'Legacy answer with an appended unresolved-information section.',
      answer_presentation: presentation,
    });

    expect(result.mode).toBe('presentation');
    expect(result.presentation).toBe(presentation);
    expect(result.legacyAnswer).toBe(
      'Legacy answer with an appended unresolved-information section.',
    );
  });

  it('uses the exact legacy answer for historical records without presentation', () => {
    const result = buildQueryAnswerViewModel({
      answer: 'Historical answer.\nSecond line.',
    });

    expect(result.mode).toBe('legacy');
    expect(result.presentation).toBeNull();
    expect(result.legacyAnswer).toBe('Historical answer.\nSecond line.');
  });

  it('presents clarification candidates even when no answer was persisted', () => {
    const result = buildQueryAnswerViewModel({
      answer: null,
      product_outcome: 'clarification_required',
      subject_scope: {
        requested_subject_ids: [], matched_subject_ids: [], catalog: [],
        document_scope: { strict: true, global: false, strict_empty: true, allowed_document_ids: [], allowed_document_count: 0 },
        source: 'inferred_project', confidence: 0.8, strict: true,
        catalog_revision: '1', policy_revision: '1', coverage_mode: 'best_evidence',
        product_outcome: 'clarification_required', clarification_reason: 'ambiguous_project_name',
        ambiguity_candidates: [
          { subject_id: 'one', kind: 'project', name: 'Orion One', aliases: [] },
          { subject_id: 'two', kind: 'project', name: 'Orion Two', aliases: [] },
        ],
        subject_lanes: [], comparison_requested: false,
      },
    });

    expect(result.mode).toBe('outcome');
    expect(result.outcome?.body).toContain('matches more than one project');
    expect(result.outcome?.candidates).toEqual(['Orion One', 'Orion Two']);
  });

  it('presents no evidence independently of a blank answer', () => {
    const result = buildQueryAnswerViewModel({ answer: null, product_outcome: 'no_evidence' });
    expect(result.mode).toBe('outcome');
    expect(result.outcome?.title).toContain('No evidence');
  });
});
