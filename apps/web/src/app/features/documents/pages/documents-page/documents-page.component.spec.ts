import { TestBed } from '@angular/core/testing';
import { Subject as Deferred, of } from 'rxjs';

import { SubjectsApiService } from '../../../subjects/data-access/subjects-api.service';
import { DocumentSubjectDecision } from '../../../subjects/models/subject.models';
import { BackgroundJobsApiService } from '../../data-access/background-jobs-api.service';
import { DocumentsApiService } from '../../data-access/documents-api.service';
import { DocumentDetail } from '../../models/document.models';
import { BackgroundJob } from '../../models/background-job.models';
import { DocumentsPageComponent } from './documents-page.component';

describe('DocumentsPageComponent', () => {
  it('ignores detail and decision responses for an older document selection', async () => {
    const alphaDocument = new Deferred<DocumentDetail>();
    const betaDocument = new Deferred<DocumentDetail>();
    const alphaDecisions = new Deferred<DocumentSubjectDecision[]>();
    const betaDecisions = new Deferred<DocumentSubjectDecision[]>();

    await TestBed.configureTestingModule({
      imports: [DocumentsPageComponent],
      providers: [
        {
          provide: DocumentsApiService,
          useValue: {
            getDocument: (id: string) => id === 'alpha' ? alphaDocument : betaDocument,
          },
        },
        {
          provide: SubjectsApiService,
          useValue: {
            listDocumentDecisions: (id: string) =>
              id === 'alpha' ? alphaDecisions : betaDecisions,
          },
        },
        { provide: BackgroundJobsApiService, useValue: {} },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(DocumentsPageComponent);
    const component = fixture.componentInstance;
    component.loadDocumentDetail('alpha');
    component.loadDocumentDetail('beta');
    expect(component.selectedDocumentDecisions()).toEqual([]);
    expect(component.documentSubjectsLoading()).toBe(true);

    betaDocument.next(makeDocument('beta'));
    betaDocument.complete();
    betaDecisions.next([makeDecision('beta', 7)]);
    betaDecisions.complete();
    alphaDocument.next(makeDocument('alpha'));
    alphaDocument.complete();
    alphaDecisions.next([makeDecision('alpha', 2)]);
    alphaDecisions.complete();

    expect(component.selectedDocument()?.id).toBe('beta');
    expect(component.selectedDocumentDecisions()[0].revision).toBe(7);
    expect(component.documentSubjectsLoading()).toBe(false);
  });

  it('keeps reclassification busy from enqueue until its tracked job is terminal', async () => {
    const reclassify = jest.fn(() => of({
      jobs: [{ job_id: 'job-1', status: 'queued', document_id: 'document-1', document_version_id: 'version-1' }],
      skipped_document_ids: [],
    }));
    const jobUpdates = new Deferred<BackgroundJob>();
    await TestBed.configureTestingModule({
      imports: [DocumentsPageComponent],
      providers: [
        {
          provide: DocumentsApiService,
          useValue: {
            getDocument: () => of(makeDocument('document-1')),
            reclassifyDocumentSubjects: reclassify,
          },
        },
        { provide: SubjectsApiService, useValue: { listDocumentDecisions: () => of([]) } },
        { provide: BackgroundJobsApiService, useValue: { getJob: () => jobUpdates } },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(DocumentsPageComponent);
    const component = fixture.componentInstance;
    component.loadDocumentDetail('document-1');

    component.reclassifySelectedDocument();
    component.reclassifySelectedDocument();

    expect(reclassify).toHaveBeenCalledTimes(1);
    expect(component.selectedDocument()?.subject_classification).toMatchObject({ status: 'queued', job_id: 'job-1' });
    expect(component.selectedClassificationBusy()).toBe(true);

    component.trackedJobs.set([makeClassificationJob('running')]);
    expect(component.selectedClassificationBusy()).toBe(true);
    component.trackedJobs.set([makeClassificationJob('succeeded')]);
    expect(component.selectedClassificationBusy()).toBe(false);
    fixture.destroy();
  });

  it.each(['failed', 'cancelled'] as const)(
    'refreshes persisted %s classification state and re-enables reclassification',
    async (terminalStatus) => {
      const initial = makeDocument('document-1');
      initial.subject_classification = classificationStatus('queued', null);
      const persisted = makeDocument('document-1');
      persisted.subject_classification = classificationStatus(
        terminalStatus,
        terminalStatus === 'failed' ? 'Classifier unavailable.' : null,
      );
      const getDocument = jest.fn()
        .mockReturnValueOnce(of(initial))
        .mockReturnValueOnce(of(persisted));
      await TestBed.configureTestingModule({
        imports: [DocumentsPageComponent],
        providers: [
          {
            provide: DocumentsApiService,
            useValue: { getDocument, listDocuments: () => of([persisted]) },
          },
          {
            provide: SubjectsApiService,
            useValue: { listDocumentDecisions: () => of([]), listSubjects: () => of([]) },
          },
          { provide: BackgroundJobsApiService, useValue: { listJobs: () => of([]) } },
        ],
      }).compileComponents();
      const fixture = TestBed.createComponent(DocumentsPageComponent);
      const component = fixture.componentInstance;
      fixture.detectChanges();
      component.loadDocumentDetail('document-1');
      expect(component.selectedClassificationBusy()).toBe(true);

      component['handleTerminalJob'](
        makeClassificationJob(
          terminalStatus,
          terminalStatus === 'failed' ? 'Classifier unavailable.' : null,
        ),
      );
      fixture.detectChanges();

      expect(getDocument).toHaveBeenCalledTimes(2);
      expect(component.selectedDocument()?.subject_classification?.status).toBe(terminalStatus);
      expect(component.selectedClassificationBusy()).toBe(false);
      expect(component.documentError()).toContain(
        terminalStatus === 'failed' ? 'Classifier unavailable.' : 'cancelled',
      );
      const button = fixture.nativeElement.querySelector(
        'button[aria-label="Reclassify document subjects"]',
      ) as HTMLButtonElement;
      expect(button.disabled).toBe(false);
    },
  );
});

function makeDocument(id: string): DocumentDetail {
  return {
    id, title: id, original_filename: `${id}.md`, content_type: 'text/markdown',
    storage_uri: null, size_bytes: 1, checksum_sha256: null, status: 'ready',
    chunk_count: 0, metadata: {}, created_at: '', updated_at: '', versions: [], chunks: [],
  };
}

function makeDecision(documentId: string, revision: number): DocumentSubjectDecision {
  return {
    id: `decision-${documentId}`, document_id: documentId, subject_id: 'subject-1',
    state: 'assigned', control_source: 'manual', confidence: null,
    confidence_band: null, rationale: null, classifier_version: null,
    policy_version: null, signals: {}, classified_document_version_id: null,
    revision, created_at: '', updated_at: '',
  };
}

function makeClassificationJob(
  status: BackgroundJob['status'],
  errorMessage: string | null = null,
): BackgroundJob {
  return {
    id: 'job-1', job_type: 'classify_document_subjects', status, priority: 0,
    payload: { document_id: 'document-1' }, result: {}, progress: status === 'succeeded' ? 1 : 0.5,
    current_stage: null, attempts: 1, max_attempts: 3, dedupe_key: null,
    scheduled_at: '', heartbeat_at: null, started_at: null, completed_at: null,
    error_message: errorMessage, created_at: '', updated_at: '',
  };
}

function classificationStatus(status: string, errorMessage: string | null) {
  return {
    status,
    job_id: 'job-1',
    document_version_id: 'version-1',
    policy_version: 'p1',
    classifier_version: 'c1',
    assigned_count: null,
    suggested_count: null,
    review_required_count: null,
    error_message: errorMessage,
  };
}
