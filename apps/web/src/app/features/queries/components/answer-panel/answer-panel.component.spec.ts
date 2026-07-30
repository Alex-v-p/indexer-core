import { TestBed } from '@angular/core/testing';

import { AnswerPresentation, QueryResponse } from '../../models/query.models';
import { buildIntegratedTraceResponse } from '../../testing/agent-trace-test.fixture';
import { AnswerPanelComponent } from './answer-panel.component';

describe('AnswerPanelComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AnswerPanelComponent],
    }).compileComponents();
  });

  it('renders a complete structured presentation with fixed sections', () => {
    const element = render(
      presentation({
        outcome: 'complete',
        title: 'Answer',
        body: 'Deployment uses a controlled rollout [1].',
        supported_information: ['Explain deployment behavior.'],
        citation_count: 1,
      }),
    );

    expect(element.querySelector('[data-testid="answer-title"]')?.textContent).toContain('Answer');
    expect(element.querySelector('[data-testid="answer-body"]')?.textContent).toContain(
      'Deployment uses a controlled rollout [1].',
    );
    expect(element.querySelector('[data-testid="supported-information"]')?.textContent).toContain(
      'Explain deployment behavior.',
    );
    expect(element.querySelector('[data-testid="unresolved-information"]')).toBeNull();
    expect(element.querySelector('[data-testid="answer-citation-count"]')?.textContent).toContain(
      '1 citations',
    );
  });

  it('renders partial unresolved disclosure once without the legacy appended section', () => {
    const result = buildIntegratedTraceResponse();
    result.answer = [
      'Deployment uses a controlled rollout [1].',
      '',
      'The available documents did not provide sufficient evidence for:',
      '- Explain rollback behavior.',
    ].join('\n');
    result.answer_presentation = presentation({
      outcome: 'partial',
      title: 'Partial answer',
      body: 'Deployment uses a controlled rollout [1].',
      supported_information: ['Explain deployment behavior.'],
      unresolved_information: ['Explain rollback behavior.'],
      citation_count: 1,
    });

    const element = renderResult(result);
    const presentationText =
      element.querySelector('[data-testid="answer-presentation"]')?.textContent ?? '';
    const unresolvedOccurrences = presentationText.match(/Explain rollback behavior\./g) ?? [];

    expect(element.querySelector('[data-testid="answer-title"]')?.textContent).toContain(
      'Partial answer',
    );
    expect(presentationText).not.toContain(
      'The available documents did not provide sufficient evidence for:',
    );
    expect(unresolvedOccurrences).toHaveLength(1);
  });

  it('renders blocked presentations without interpreting the body as HTML', () => {
    const element = render(
      presentation({
        outcome: 'blocked_no_evidence',
        title: 'No evidence available',
        body: '<strong>No evidence was found.</strong>',
      }),
    );

    expect(
      element.querySelector('[data-testid="answer-presentation"]')?.getAttribute('data-outcome'),
    ).toBe('blocked_no_evidence');
    expect(element.querySelector('[data-testid="answer-body"]')?.textContent).toContain(
      '<strong>No evidence was found.</strong>',
    );
    expect(element.querySelector('[data-testid="answer-body"] strong')).toBeNull();
  });

  it('falls back exactly to the legacy answer when presentation is absent', () => {
    const result = buildIntegratedTraceResponse();
    result.answer = 'Legacy answer with its original formatting.\nSecond line.';
    delete result.answer_presentation;

    const element = renderResult(result);

    expect(element.querySelector('[data-testid="answer-presentation"]')).toBeNull();
    expect(element.querySelector('[data-testid="legacy-answer"]')?.textContent?.trim()).toBe(
      result.answer,
    );
  });

  function render(value: AnswerPresentation): HTMLElement {
    const result = buildIntegratedTraceResponse();
    result.answer_presentation = value;
    return renderResult(result);
  }

  function renderResult(result: QueryResponse): HTMLElement {
    const fixture = TestBed.createComponent(AnswerPanelComponent);
    fixture.componentInstance.result = result;
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }
});

function presentation(
  overrides: Partial<AnswerPresentation> & Pick<AnswerPresentation, 'outcome' | 'title' | 'body'>,
): AnswerPresentation {
  return {
    schema_version: '1.0',
    supported_information: [],
    unresolved_information: [],
    citation_count: 0,
    ...overrides,
  };
}
