import { TestBed } from '@angular/core/testing';

import { CitationListComponent } from './citation-list.component';

describe('CitationListComponent subject lanes', () => {
  it('badges lane-attributed citations and keeps legacy citations readable', async () => {
    await TestBed.configureTestingModule({ imports: [CitationListComponent] }).compileComponents();
    const fixture = TestBed.createComponent(CitationListComponent);
    fixture.componentInstance.citations = [
      { id: null, citation_index: 1, label: null, evidence_id: null, page_number: null, quote: 'Lane quote', qdrant_chunk_index_id: null, document_id: null, document_version_id: null, subject_name: 'Orion', metadata: {} },
      { id: null, citation_index: 2, label: null, evidence_id: null, page_number: null, quote: 'Legacy quote', qdrant_chunk_index_id: null, document_id: null, document_version_id: null, metadata: {} },
    ];
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Orion');
    expect(fixture.nativeElement.textContent).toContain('Legacy quote');
  });
});
