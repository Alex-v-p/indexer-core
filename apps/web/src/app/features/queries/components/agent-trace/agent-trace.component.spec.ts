import { TestBed } from '@angular/core/testing';

import {
  buildIntegratedTraceResponse,
  buildPartialTraceResponse,
} from '../../testing/agent-trace-test.fixture';
import { InformationNeedAttempt } from '../../models/query.models';
import { AgentTraceComponent } from './agent-trace.component';

describe('AgentTraceComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AgentTraceComponent],
    }).compileComponents();
  });

  it('renders the complete eight-section hierarchy in the required order', () => {
    const fixture = TestBed.createComponent(AgentTraceComponent);
    fixture.componentInstance.result = buildIntegratedTraceResponse();
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    const sections = Array.from(
      element.querySelectorAll<HTMLElement>('[data-testid="trace-section"]'),
    );
    const sectionOrder = sections.map((section) => section.dataset['sectionOrder']);
    const sectionText = sections.map((section) => section.textContent ?? '');

    expect(sectionOrder).toEqual(['1', '2', '3', '4', '5', '6', '7', '8']);
    expect(sectionText[0]).toContain('Agent Journey');
    expect(sectionText[1]).toContain('Query classification');
    expect(sectionText[2]).toContain('Information needs');
    expect(sectionText[3]).toContain('Retrieval strategy evolution');
    expect(sectionText[4]).toContain('Primary-document preference and balancing');
    expect(sectionText[5]).toContain('Final evidence arbitration');
    expect(sectionText[6]).toContain('Detailed information-need execution');
    expect(sectionText[7]).toContain('Runtime node timeline');
  });

  it('keeps retries inside need lanes and low-level diagnostics collapsed', () => {
    const fixture = TestBed.createComponent(AgentTraceComponent);
    fixture.componentInstance.result = buildIntegratedTraceResponse();
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    const runtimeSections = Array.from(
      element.querySelectorAll<HTMLDetailsElement>('[data-detail-level="runtime"]'),
    );

    expect(element.querySelector('app-retrieval-retry-trace')).toBeNull();
    expect(element.textContent).not.toContain('Retries and fallbacks');
    expect(element.querySelectorAll('[data-testid="retry-transition"]')).toHaveLength(1);
    expect(runtimeSections).toHaveLength(2);
    expect(runtimeSections.every((section) => !section.open)).toBe(true);
    expect(element.querySelector('[aria-label="Agent decision flow"]')).toBeNull();
  });

  it('renders a partial older response safely and omits unavailable sections', () => {
    const fixture = TestBed.createComponent(AgentTraceComponent);

    expect(() => {
      fixture.componentInstance.result = buildPartialTraceResponse();
      fixture.detectChanges();
    }).not.toThrow();

    const element = fixture.nativeElement as HTMLElement;
    const sectionOrder = Array.from(
      element.querySelectorAll<HTMLElement>('[data-testid="trace-section"]'),
    ).map((section) => section.dataset['sectionOrder']);

    expect(sectionOrder).toEqual(['1']);
    expect(element.textContent).toContain(
      'This run predates structured information-need resolution.',
    );
  });

  it('recovers information needs from resolution when decomposition is absent', () => {
    const fixture = TestBed.createComponent(AgentTraceComponent);
    const result = buildIntegratedTraceResponse();
    result.information_need_decomposition = null;
    fixture.componentInstance.result = result;
    fixture.detectChanges();

    const needSection = fixture.nativeElement.querySelector(
      '[data-section-order="3"]',
    ) as HTMLElement | null;

    expect(needSection?.textContent).toContain('Explain deployment behavior');
    expect(needSection?.textContent).toContain('need-1');
  });

  it('degrades safely when an older attempt lacks snapshots and retrieval metadata', () => {
    const fixture = TestBed.createComponent(AgentTraceComponent);
    const result = buildIntegratedTraceResponse();
    const execution = result.information_need_resolution!.executions[0];
    const plan = execution.current_plan!;
    const attemptGrading = result.evidence_grading!;
    plan.preferred_document = null;
    for (const historicalPlan of execution.plan_history) {
      historicalPlan.preferred_document = null;
    }
    execution.attempts = [
      {
        attempt_number: 1,
        query: plan.query,
        top_k: plan.top_k,
        pipeline_name: plan.selected_pipeline_name,
        strategy: plan.strategy,
        adjustments: [],
        retrieved_count: 2,
        unique_evidence_added: 1,
        evidence_keys: [],
        plan,
        constraint_validation: {
          status: 'not_requested',
          blocked: false,
          candidate_count: 2,
          matched_count: 2,
          rejected_count: 0,
          constraints: {},
          rationale: 'No constraints were requested.',
        },
        evidence_grading: attemptGrading,
      } as unknown as InformationNeedAttempt,
    ];
    execution.plan_history = [plan];
    execution.attempts_used = 1;
    result.primary_document_preference = null;
    result.retrieval_retry = null;
    result.evidence_grading = null;
    fixture.componentInstance.result = result;

    expect(() => fixture.detectChanges()).not.toThrow();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.textContent).toContain(
      'Evidence snapshots were not stored for this older run.',
    );
    expect(element.querySelector('[data-section-order="6"]')).toBeNull();
    expect(element.querySelector('app-retrieval-retry-trace')).toBeNull();
  });
});
