import { TestBed } from '@angular/core/testing';

import { EvidenceViewerComponent } from './evidence-viewer.component';

describe('EvidenceViewerComponent subject lanes', () => {
  it('badges lane-attributed evidence without requiring lane fields on legacy evidence', async () => {
    await TestBed.configureTestingModule({ imports: [EvidenceViewerComponent] }).compileComponents();
    const fixture = TestBed.createComponent(EvidenceViewerComponent);
    fixture.componentInstance.evidence = [
      { id: null, rank: 1, score: 1, text: 'Lane evidence', qdrant_chunk_index_id: null, document_id: null, document_version_id: null, subject_name: 'Orion', metadata: {} },
      { id: null, rank: 2, score: null, text: 'Legacy evidence', qdrant_chunk_index_id: null, document_id: null, document_version_id: null, metadata: {} },
    ];
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Orion');
    expect(fixture.nativeElement.textContent).toContain('Legacy evidence');
  });
});
