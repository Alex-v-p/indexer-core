import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { DocumentsApiService } from './documents-api.service';

describe('DocumentsApiService', () => {
  it('sends every selected subject with a single upload', () => {
    TestBed.configureTestingModule({
      providers: [DocumentsApiService, provideHttpClient(), provideHttpClientTesting()],
    });
    const service = TestBed.inject(DocumentsApiService);
    const http = TestBed.inject(HttpTestingController);
    const file = new File(['content'], 'notes.md', { type: 'text/markdown' });

    service
      .uploadDocuments({
        files: [file],
        detectExistingVersions: true,
        subjectIds: ['subject-1', 'subject-2'],
      })
      .subscribe();

    const request = http.expectOne('/documents');
    const form = request.request.body as FormData;
    expect(form.getAll('subject_ids')).toEqual(['subject-1', 'subject-2']);
    request.flush({}, { headers: { 'X-Background-Job-ID': 'job-1' } });
    http.verify();
  });

  it('queues reclassification and a bounded classification backfill', () => {
    TestBed.configureTestingModule({
      providers: [DocumentsApiService, provideHttpClient(), provideHttpClientTesting()],
    });
    const service = TestBed.inject(DocumentsApiService);
    const http = TestBed.inject(HttpTestingController);

    service.reclassifyDocumentSubjects('document-1').subscribe();
    const reclassify = http.expectOne('/documents/document-1/subject-classification');
    expect(reclassify.request.method).toBe('POST');
    reclassify.flush({ jobs: [], skipped_document_ids: [] });

    service.backfillSubjectClassification(25).subscribe();
    const backfill = http.expectOne(
      (request) => request.url === '/documents/subject-classification/backfill' && request.params.get('limit') === '25',
    );
    expect(backfill.request.method).toBe('POST');
    backfill.flush({ jobs: [], skipped_document_ids: [] });
    http.verify();
  });
});
