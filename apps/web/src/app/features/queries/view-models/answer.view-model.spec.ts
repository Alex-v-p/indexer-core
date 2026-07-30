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
});
