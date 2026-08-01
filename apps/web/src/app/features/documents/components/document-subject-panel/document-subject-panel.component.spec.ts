import { DocumentSubjectPanelComponent } from './document-subject-panel.component';
import { TestBed } from '@angular/core/testing';

describe('DocumentSubjectPanelComponent', () => {
  it('uses revision zero for a new manual assignment', () => {
    const component = new DocumentSubjectPanelComponent();
    const emitted: unknown[] = [];
    component.decisionRequested.subscribe((request) => emitted.push(request));

    component.requestState('subject-1', 'assigned');

    expect(emitted).toEqual([
      { subjectId: 'subject-1', state: 'assigned', expectedRevision: 0 },
    ]);
  });

  it('reviews a suggestion using its current revision', () => {
    const component = new DocumentSubjectPanelComponent();
    const emitted: unknown[] = [];
    component.suggestionReviewRequested.subscribe((request) => emitted.push(request));

    component.reviewSuggestion(
      {
        id: 'decision-1',
        document_id: 'document-1',
        subject_id: 'subject-1',
        state: 'suggested',
        control_source: 'automatic',
        confidence: 0.7,
        confidence_band: 'medium',
        rationale: null,
        classifier_version: 'v1',
        policy_version: 'v1',
        signals: {},
        classified_document_version_id: 'version-1',
        revision: 5,
        created_at: '2026-08-01T00:00:00Z',
        updated_at: '2026-08-01T00:00:00Z',
      },
      'accept',
    );

    expect(emitted).toEqual([
      { subjectId: 'subject-1', decision: 'accept', expectedRevision: 5 },
    ]);
  });

  it('disables actions while loading and gives repeated actions target-specific labels', async () => {
    await TestBed.configureTestingModule({ imports: [DocumentSubjectPanelComponent] })
      .compileComponents();
    const fixture = TestBed.createComponent(DocumentSubjectPanelComponent);
    fixture.componentInstance.subjects = [
      {
        id: 'subject-1', kind: 'project', name: 'Orion', normalized_name: 'orion',
        description: null, metadata: {}, created_at: '', updated_at: '',
        archived_at: null, aliases: [],
      },
    ];
    fixture.componentInstance.loading = true;
    fixture.detectChanges();

    const assign = fixture.nativeElement.querySelector(
      'button[aria-label="Assign Orion"]',
    ) as HTMLButtonElement;
    expect(assign.disabled).toBe(true);
    expect(fixture.nativeElement.querySelector('[role="status"]')?.textContent)
      .toContain('Loading subject assignments');
  });

  it('shows classifier status and exposes an accessible reclassify action', async () => {
    await TestBed.configureTestingModule({ imports: [DocumentSubjectPanelComponent] })
      .compileComponents();
    const fixture = TestBed.createComponent(DocumentSubjectPanelComponent);
    fixture.componentInstance.classificationStatus = {
      status: 'succeeded', job_id: 'job-1', document_version_id: 'version-1',
      policy_version: 'p1', classifier_version: 'c1', assigned_count: 2,
      suggested_count: 1, review_required_count: 1, error_message: null,
    };
    const emitted: void[] = [];
    fixture.componentInstance.reclassifyRequested.subscribe(() => emitted.push(undefined));
    fixture.detectChanges();

    const button = fixture.nativeElement.querySelector(
      'button[aria-label="Reclassify document subjects"]',
    ) as HTMLButtonElement;
    expect(fixture.nativeElement.textContent).toContain('2 assigned');
    expect(button.disabled).toBe(false);
    fixture.componentInstance.requestReclassification();
    expect(emitted).toHaveLength(1);
  });
});
