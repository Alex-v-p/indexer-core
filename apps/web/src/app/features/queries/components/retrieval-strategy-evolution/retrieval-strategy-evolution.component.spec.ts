import { TestBed } from '@angular/core/testing';

import { buildIntegratedTraceResponse } from '../../testing/agent-trace-test.fixture';
import { RetrievalStrategyEvolutionComponent } from './retrieval-strategy-evolution.component';

describe('RetrievalStrategyEvolutionComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RetrievalStrategyEvolutionComponent],
    }).compileComponents();
  });

  it('renders ordered strategy history, adjustments, change reasons, and the initial plan', () => {
    const fixture = TestBed.createComponent(RetrievalStrategyEvolutionComponent);
    fixture.componentInstance.result = buildIntegratedTraceResponse();
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    const attempts = Array.from(
      element.querySelectorAll('[data-testid="strategy-evolution-attempt"]'),
    ).map((item) => item.textContent ?? '');
    const text = element.textContent ?? '';

    expect(attempts).toHaveLength(2);
    expect(attempts[0]).toContain('Attempt 1: hybrid → hybrid-rag');
    expect(attempts[1]).toContain('Attempt 2: multi query → multi-query-rag');
    expect(attempts[1]).toContain('broaden query');
    expect(element.querySelector('[data-testid="strategy-change-reason"]')?.textContent).toContain(
      'switched hybrid to multi query',
    );
    expect(text).toContain('Initial query-level plan');
    expect(text).not.toContain('Selected retrieval plan');
  });
});
