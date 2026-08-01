import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { SubjectsApiService } from './subjects-api.service';

describe('SubjectsApiService', () => {
  let service: SubjectsApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [SubjectsApiService, provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SubjectsApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('writes revision-aware manual decisions', () => {
    service.setDocumentDecision('document-1', 'subject-1', 'rejected', 4).subscribe();

    const request = http.expectOne('/documents/document-1/subjects/subject-1');
    expect(request.request.method).toBe('PUT');
    expect(request.request.body).toEqual({
      state: 'rejected',
      expected_revision: 4,
      rationale: undefined,
    });
    request.flush({});
  });

  it('reviews automatic suggestions with their current revision', () => {
    service.reviewSuggestion('document-1', 'subject-1', 'accept', 3).subscribe();

    const request = http.expectOne(
      '/documents/document-1/subject-suggestions/subject-1/review',
    );
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({ decision: 'accept', expected_revision: 3 });
    request.flush({});
  });
});
