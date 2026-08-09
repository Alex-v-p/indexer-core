import { HttpErrorResponse } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { NEVER, Subject as Deferred, of, throwError } from 'rxjs';

import { DocumentOrganizationApiService } from '../../../organization/data-access/document-organization-api.service';
import { DocumentOrganization } from '../../../organization/models/document-organization.models';
import { BackgroundJobsApiService } from '../../data-access/background-jobs-api.service';
import { DocumentsApiService } from '../../data-access/documents-api.service';
import { BackgroundJob } from '../../models/background-job.models';
import { DocumentDetail } from '../../models/document.models';
import { DocumentsPageComponent } from './documents-page.component';

describe('DocumentsPageComponent', () => {
  it('ignores document organization responses for an older selection', async () => {
    const alphaDocument = new Deferred<DocumentDetail>();
    const betaDocument = new Deferred<DocumentDetail>();
    const alphaOrganization = new Deferred<DocumentOrganization>();
    const betaOrganization = new Deferred<DocumentOrganization>();

    await TestBed.configureTestingModule({
      imports: [DocumentsPageComponent],
      providers: [
        {
          provide: DocumentsApiService,
          useValue: { getDocument: (id: string) => id === 'alpha' ? alphaDocument : betaDocument },
        },
        {
          provide: DocumentOrganizationApiService,
          useValue: {
            getDocumentOrganization: (id: string) =>
              id === 'alpha' ? alphaOrganization : betaOrganization,
          },
        },
        { provide: BackgroundJobsApiService, useValue: {} },
      ],
    }).compileComponents();

    const component = TestBed.createComponent(DocumentsPageComponent).componentInstance;
    component.loadDocumentDetail('alpha');
    component.loadDocumentDetail('beta');

    betaDocument.next(makeDocument('beta'));
    betaDocument.complete();
    betaOrganization.next(makeOrganization('beta', 7));
    betaOrganization.complete();
    alphaDocument.next(makeDocument('alpha'));
    alphaDocument.complete();
    alphaOrganization.next(makeOrganization('alpha', 2));
    alphaOrganization.complete();

    expect(component.selectedDocument()?.id).toBe('beta');
    expect(component.selectedDocumentOrganization()?.content_group_assignment?.revision).toBe(7);
    expect(component.documentOrganizationLoading()).toBe(false);
  });

  it('refreshes and explains a CAS conflict', async () => {
    const getOrganization = jest.fn()
      .mockReturnValueOnce(of(makeOrganization('document-1', 2)))
      .mockReturnValueOnce(of(makeOrganization('document-1', 3)));
    const setGroup = jest.fn(() => throwError(() => new HttpErrorResponse({
      status: 409,
      statusText: 'Conflict',
      error: { detail: { message: 'stale group assignment' } },
    })));
    await TestBed.configureTestingModule({
      imports: [DocumentsPageComponent],
      providers: [
        { provide: DocumentsApiService, useValue: { getDocument: () => of(makeDocument('document-1')) } },
        {
          provide: DocumentOrganizationApiService,
          useValue: { getDocumentOrganization: getOrganization, setDocumentContentGroup: setGroup },
        },
        { provide: BackgroundJobsApiService, useValue: {} },
      ],
    }).compileComponents();
    const component = TestBed.createComponent(DocumentsPageComponent).componentInstance;
    component.loadDocumentDetail('document-1');

    component.setDocumentContentGroup({ contentGroupId: 'group-2', expectedRevision: 2 });

    expect(getOrganization).toHaveBeenCalledTimes(2);
    expect(component.selectedDocumentOrganization()?.content_group_assignment?.revision).toBe(3);
    expect(component.documentError()).toContain('latest organization has been loaded');
  });

  it('keeps reclassification busy until its tracked organization job is terminal', async () => {
    const requeue = jest.fn(() => of({ job_ids: ['job-1'], skipped_document_ids: [] }));
    const jobUpdates = new Deferred<BackgroundJob>();
    await TestBed.configureTestingModule({
      imports: [DocumentsPageComponent],
      providers: [
        { provide: DocumentsApiService, useValue: { getDocument: () => of(makeDocument('document-1')) } },
        {
          provide: DocumentOrganizationApiService,
          useValue: {
            getDocumentOrganization: () => of(makeOrganization('document-1', 1)),
            requeueDocument: requeue,
          },
        },
        { provide: BackgroundJobsApiService, useValue: { getJob: () => jobUpdates } },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(DocumentsPageComponent);
    const component = fixture.componentInstance;
    component.loadDocumentDetail('document-1');

    component.reclassifySelectedDocument();
    component.reclassifySelectedDocument();

    expect(requeue).toHaveBeenCalledTimes(1);
    expect(component.selectedClassificationBusy()).toBe(true);
    component.trackedJobs.set([makeClassificationJob('succeeded')]);
    expect(component.selectedClassificationBusy()).toBe(false);
    fixture.destroy();
  });

  it('auto-tracks an organization child job discovered after ingestion reload', async () => {
    const queuedOrganization = makeOrganization(
      'document-1',
      2,
      'queued',
      'organization-job-1',
    );
    const getOrganization = jest.fn()
      .mockReturnValueOnce(of(makeOrganization('document-1', 1)))
      .mockReturnValueOnce(of(queuedOrganization));
    await TestBed.configureTestingModule({
      imports: [DocumentsPageComponent],
      providers: [
        {
          provide: DocumentsApiService,
          useValue: {
            getDocument: () => of(makeDocument('document-1')),
            listDocuments: () => of([makeDocument('document-1')]),
          },
        },
        {
          provide: DocumentOrganizationApiService,
          useValue: { getDocumentOrganization: getOrganization },
        },
        {
          provide: BackgroundJobsApiService,
          useValue: { getJob: () => NEVER },
        },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(DocumentsPageComponent);
    const component = fixture.componentInstance;
    component.loadDocumentDetail('document-1');

    component['handleTerminalJob'](makeDocumentJob('ingest_document', 'succeeded'));

    expect(getOrganization).toHaveBeenCalledTimes(2);
    expect(component['jobPolling'].has('organization-job-1')).toBe(true);
    expect(component.selectedClassificationBusy()).toBe(true);

    component['acceptDocumentOrganization'](queuedOrganization);
    expect(component['jobPolling'].size).toBe(1);

    component.trackedJobs.set([
      makeDocumentJob('classify_document_organization', 'succeeded', 'organization-job-1'),
    ]);
    expect(component.selectedClassificationBusy()).toBe(false);
    fixture.destroy();
  });

  it('resumes active legacy and organization classification jobs', async () => {
    const legacy = makeDocumentJob('classify_document_subjects', 'running', 'legacy-job');
    const organization = makeDocumentJob(
      'classify_document_organization',
      'queued',
      'organization-job',
    );
    await TestBed.configureTestingModule({
      imports: [DocumentsPageComponent],
      providers: [
        {
          provide: DocumentsApiService,
          useValue: { listDocuments: () => of([]) },
        },
        {
          provide: DocumentOrganizationApiService,
          useValue: {
            listContentGroups: () => of([]),
            listDocumentTypes: () => of([]),
          },
        },
        {
          provide: BackgroundJobsApiService,
          useValue: {
            listJobs: () => of([legacy, organization]),
            getJob: () => NEVER,
          },
        },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(DocumentsPageComponent);

    fixture.detectChanges();

    expect(componentPollingIds(fixture.componentInstance)).toEqual([
      'legacy-job',
      'organization-job',
    ]);
    fixture.destroy();
  });
});

function componentPollingIds(component: DocumentsPageComponent): string[] {
  return [...component['jobPolling'].keys()].sort();
}

function makeDocument(id: string): DocumentDetail {
  return {
    id, title: id, original_filename: `${id}.md`, content_type: 'text/markdown',
    storage_uri: null, size_bytes: 1, checksum_sha256: null, status: 'ready',
    chunk_count: 0, metadata: {}, created_at: '', updated_at: '', versions: [], chunks: [],
  };
}

function makeOrganization(
  documentId: string,
  revision: number,
  status = 'succeeded',
  jobId: string | null = null,
): DocumentOrganization {
  return {
    document_id: documentId,
    type_decisions: [],
    content_group_assignment: {
      document_id: documentId, content_group_id: 'group-1', state: 'assigned',
      source: 'manual', unresolved_reason: null, confidence: null, confidence_band: null,
      rationale: null, classifier_version: null, policy_version: null, signals: {},
      summary_hash: null, classified_document_version_id: null, revision,
      created_at: '', updated_at: '',
    },
    content_group: null,
    status: { status, job_id: jobId },
  };
}

function makeClassificationJob(status: BackgroundJob['status']): BackgroundJob {
  return makeDocumentJob('classify_document_organization', status);
}

function makeDocumentJob(
  jobType: string,
  status: BackgroundJob['status'],
  id = 'job-1',
): BackgroundJob {
  return {
    id, job_type: jobType, status, priority: 0,
    payload: { document_id: 'document-1' }, result: {},
    progress: status === 'succeeded' ? 1 : 0.5, current_stage: null,
    attempts: 1, max_attempts: 3, dedupe_key: null, scheduled_at: '',
    heartbeat_at: null, started_at: null, completed_at: null, error_message: null,
    created_at: '', updated_at: '',
  };
}
