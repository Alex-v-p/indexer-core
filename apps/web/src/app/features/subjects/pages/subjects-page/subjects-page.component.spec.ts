import { TestBed } from '@angular/core/testing';
import { Subject as Deferred, of } from 'rxjs';

import { DocumentsApiService } from '../../../documents/data-access/documents-api.service';
import { BackgroundJobsApiService } from '../../../../core/background-jobs/background-jobs-api.service';
import { SubjectsApiService } from '../../data-access/subjects-api.service';
import { DocumentSubjectDecision, Subject } from '../../models/subject.models';
import { SubjectsPageComponent } from './subjects-page.component';

describe('SubjectsPageComponent', () => {
  it('loads the subject administration workspace', async () => {
    await TestBed.configureTestingModule({
      imports: [SubjectsPageComponent],
      providers: [
        {
          provide: SubjectsApiService,
          useValue: { listSubjects: () => of([]) },
        },
        {
          provide: DocumentsApiService,
          useValue: { listDocuments: () => of([]) },
        },
        { provide: BackgroundJobsApiService, useValue: { getJob: jest.fn() } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(SubjectsPageComponent);
    fixture.detectChanges();

    expect(fixture.componentInstance.subjects()).toEqual([]);
    expect(fixture.nativeElement.querySelector('h1').textContent).toContain('Subjects');
  });

  it('ignores an older subject membership response and uses the current revision', async () => {
    const alphaResponse = new Deferred<DocumentSubjectDecision[]>();
    const betaResponse = new Deferred<DocumentSubjectDecision[]>();
    const setDocumentDecision = jest.fn(() => of({}));
    let membershipCall = 0;
    const subjects = [makeSubject('alpha', 'Alpha'), makeSubject('beta', 'Beta')];

    await TestBed.configureTestingModule({
      imports: [SubjectsPageComponent],
      providers: [
        {
          provide: SubjectsApiService,
          useValue: {
            listSubjects: () => of(subjects),
            listDocumentDecisions: () =>
              membershipCall++ === 0 ? alphaResponse : betaResponse,
            setDocumentDecision,
          },
        },
        {
          provide: DocumentsApiService,
          useValue: { listDocuments: () => of([makeDocument()]) },
        },
        { provide: BackgroundJobsApiService, useValue: { getJob: jest.fn() } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(SubjectsPageComponent);
    fixture.detectChanges();
    fixture.componentInstance.selectSubject(subjects[0]);
    fixture.componentInstance.selectSubject(subjects[1]);

    expect(fixture.componentInstance.memberships()).toEqual([]);
    expect(fixture.componentInstance.membershipsLoading()).toBe(true);

    betaResponse.next([makeDecision('beta', 'rejected', 8)]);
    betaResponse.complete();
    alphaResponse.next([makeDecision('alpha', 'assigned', 2)]);
    alphaResponse.complete();
    fixture.detectChanges();

    expect(fixture.componentInstance.memberships()[0].decision?.subject_id).toBe('beta');
    const assign = fixture.nativeElement.querySelector(
      'button[aria-label="Assign Beta to Architecture"]',
    ) as HTMLButtonElement;
    expect(assign).not.toBeNull();
    assign.click();
    expect(setDocumentDecision).toHaveBeenCalledWith(
      'document-1', 'beta', 'assigned', 8,
    );
    expect(
      fixture.nativeElement.querySelector('label[for="subject-alias-name"]')?.textContent,
    ).toContain('Add alias');
  });

  it('bounds and queues the automatic-classification backfill', async () => {
    const backfill = jest.fn(() => of({ jobs: [], skipped_document_ids: ['document-1'] }));
    await TestBed.configureTestingModule({
      imports: [SubjectsPageComponent],
      providers: [
        { provide: SubjectsApiService, useValue: { listSubjects: () => of([]) } },
        { provide: DocumentsApiService, useValue: { listDocuments: () => of([]), backfillSubjectClassification: backfill } },
        { provide: BackgroundJobsApiService, useValue: { getJob: jest.fn() } },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(SubjectsPageComponent);
    fixture.componentInstance.backfillLimit = 500;

    fixture.componentInstance.startClassificationBackfill();
    fixture.detectChanges();

    expect(backfill).toHaveBeenCalledWith(100);
    expect(fixture.nativeElement.querySelector('[role="status"]')?.textContent).toContain('already current');
  });
});

function makeSubject(id: string, name: string): Subject {
  return {
    id, kind: 'project', name, normalized_name: name.toLowerCase(),
    description: null, metadata: {}, created_at: '', updated_at: '',
    archived_at: null, aliases: [],
  };
}

function makeDocument() {
  return {
    id: 'document-1', title: 'Architecture', original_filename: 'architecture.md',
    content_type: 'text/markdown', storage_uri: null, size_bytes: 1,
    checksum_sha256: null, status: 'ready', chunk_count: 0, metadata: {},
    created_at: '', updated_at: '',
  };
}

function makeDecision(
  subjectId: string,
  state: DocumentSubjectDecision['state'],
  revision: number,
): DocumentSubjectDecision {
  return {
    id: `decision-${subjectId}`, document_id: 'document-1', subject_id: subjectId,
    state, control_source: 'manual', confidence: null, confidence_band: null,
    rationale: null, classifier_version: null, policy_version: null, signals: {},
    classified_document_version_id: null, revision, created_at: '', updated_at: '',
  };
}
