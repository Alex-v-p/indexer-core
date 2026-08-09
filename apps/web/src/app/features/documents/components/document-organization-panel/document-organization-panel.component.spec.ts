import { TestBed } from '@angular/core/testing';

import { DocumentOrganizationPanelComponent } from './document-organization-panel.component';

describe('DocumentOrganizationPanelComponent', () => {
  it('emits group changes and clear operations with the current revision', () => {
    const component = new DocumentOrganizationPanelComponent();
    const emitted: unknown[] = [];
    component.groupChangeRequested.subscribe((value) => emitted.push(value));
    component.organization = makeOrganization();

    component.selectedGroupId = 'group-2';
    component.saveGroup();
    component.clearGroup();

    expect(emitted).toEqual([
      { contentGroupId: 'group-2', expectedRevision: 4 },
      { contentGroupId: null, expectedRevision: 4 },
    ]);
  });

  it('emits only changed type decisions with explicit revisions', () => {
    const component = new DocumentOrganizationPanelComponent();
    component.documentTypes = [makeType('type-1'), makeType('type-2')];
    component.organization = makeOrganization();
    const emitted: unknown[] = [];
    component.typesChangeRequested.subscribe((value) => emitted.push(value));

    component.toggleType('type-1', false);
    component.toggleType('type-2', true);
    component.saveTypes();

    expect(emitted).toEqual([{ decisions: [
      { document_type_id: 'type-1', state: 'rejected', expected_revision: 3 },
      { document_type_id: 'type-2', state: 'assigned', expected_revision: 0 },
    ] }]);
  });

  it('renders pending provenance and accessible reclassification controls', async () => {
    await TestBed.configureTestingModule({ imports: [DocumentOrganizationPanelComponent] })
      .compileComponents();
    const fixture = TestBed.createComponent(DocumentOrganizationPanelComponent);
    fixture.componentInstance.organization = makeOrganization('pending');
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('pending');
    expect(fixture.nativeElement.textContent).toContain('automatic');
    expect(fixture.nativeElement.querySelector(
      'button[aria-label="Reclassify document organization"]',
    )).not.toBeNull();
  });
});

function makeType(id: string) {
  return {
    id, key: id, label: id, description: null, metadata: {}, archived_at: null,
    created_at: '', updated_at: '',
  };
}

function makeOrganization(state: 'pending' | 'assigned' = 'assigned') {
  return {
    document_id: 'document-1',
    type_decisions: [{
      document_type: makeType('type-1'),
      decision: {
        id: 'decision-1', document_id: 'document-1', document_type_id: 'type-1',
        state: 'assigned' as const, source: 'manual' as const, confidence: null,
        confidence_band: null, rationale: null, classifier_version: null,
        policy_version: null, signals: {}, classified_document_version_id: null,
        revision: 3, created_at: '', updated_at: '',
      },
    }],
    content_group_assignment: {
      document_id: 'document-1', content_group_id: state === 'assigned' ? 'group-1' : null,
      state, source: state === 'assigned' ? 'manual' as const : 'automatic' as const,
      unresolved_reason: null, confidence: null, confidence_band: null, rationale: null,
      classifier_version: null, policy_version: null, signals: {}, summary_hash: null,
      classified_document_version_id: null, revision: 4, created_at: '', updated_at: '',
    },
    content_group: state === 'assigned' ? {
      id: 'group-1', name: 'DAF', normalized_name: 'daf', description: null,
      metadata: {}, archived_at: null, created_at: '', updated_at: '', aliases: [],
    } : null,
    status: { status: 'succeeded' },
  };
}
