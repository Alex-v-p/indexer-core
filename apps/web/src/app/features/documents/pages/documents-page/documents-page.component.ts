import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { Subscription, finalize, forkJoin, switchMap, takeWhile, timer } from 'rxjs';

import { toApiErrorMessage } from '../../../../core/http/api-error';
import { DocumentJobProgressComponent } from '../../components/document-job-progress/document-job-progress.component';
import {
  DocumentBatchSelectionChange,
  DocumentListComponent,
} from '../../components/document-list/document-list.component';
import {
  DeleteDocumentVersionsRequest,
  DocumentMetadataPanelComponent,
} from '../../components/document-metadata-panel/document-metadata-panel.component';
import { DocumentUploadComponent } from '../../components/document-upload/document-upload.component';
import {
  DocumentSubjectChangeRequest,
  DocumentSubjectPanelComponent,
  DocumentSuggestionReviewRequest,
} from '../../components/document-subject-panel/document-subject-panel.component';
import { BackgroundJobsApiService } from '../../data-access/background-jobs-api.service';
import { DocumentsApiService } from '../../data-access/documents-api.service';
import { BackgroundJob } from '../../models/background-job.models';
import { DocumentDetail, DocumentSummary, DocumentUploadRequest } from '../../models/document.models';
import { SubjectsApiService } from '../../../subjects/data-access/subjects-api.service';
import { DocumentSubjectDecision, Subject } from '../../../subjects/models/subject.models';

@Component({
  selector: 'app-documents-page',
  standalone: true,
  imports: [
    DocumentUploadComponent,
    DocumentListComponent,
    DocumentMetadataPanelComponent,
    DocumentJobProgressComponent,
    DocumentSubjectPanelComponent,
  ],
  templateUrl: './documents-page.component.html',
})
export class DocumentsPageComponent implements OnInit, OnDestroy {
  private readonly documentsApi = inject(DocumentsApiService);
  private readonly jobsApi = inject(BackgroundJobsApiService);
  private readonly subjectsApi = inject(SubjectsApiService);
  private readonly jobPolling = new Map<string, Subscription>();
  private activeDocumentId: string | null = null;
  private documentDetailRequest = 0;
  private documentSubjectsRequest = 0;

  readonly documents = signal<DocumentSummary[]>([]);
  readonly selectedDocument = signal<DocumentDetail | null>(null);
  readonly subjects = signal<Subject[]>([]);
  readonly selectedDocumentDecisions = signal<DocumentSubjectDecision[]>([]);
  readonly batchSelectedDocumentIds = signal<string[]>([]);
  readonly trackedJobs = signal<BackgroundJob[]>([]);
  readonly documentsLoading = signal(false);
  readonly documentUploading = signal(false);
  readonly deletionSubmitting = signal(false);
  readonly subjectMutationBusy = signal(false);
  readonly classificationSubmitting = signal(false);
  readonly documentSubjectsLoading = signal(false);
  readonly documentError = signal<string | null>(null);
  readonly versionDeletionBusy = computed(
    () =>
      this.deletionSubmitting() ||
      this.trackedJobs().some(
        (job) => job.job_type === 'delete_document_versions' && isRunningJob(job),
      ),
  );
  readonly selectedClassificationBusy = computed(() => {
    if (this.classificationSubmitting()) {
      return true;
    }
    const document = this.selectedDocument();
    if (!document) {
      return false;
    }
    const status = document.subject_classification;
    const tracked = status?.job_id
      ? this.trackedJobs().find((job) => job.id === status.job_id)
      : undefined;
    if (tracked) {
      return isRunningJob(tracked);
    }
    return status?.status === 'queued' || status?.status === 'running';
  });

  ngOnInit(): void {
    this.loadDocuments();
    this.loadSubjects();
    this.resumeActiveDocumentJobs();
  }

  ngOnDestroy(): void {
    for (const subscription of this.jobPolling.values()) {
      subscription.unsubscribe();
    }
  }

  loadDocuments(): void {
    this.documentsLoading.set(true);
    this.documentError.set(null);

    this.documentsApi
      .listDocuments()
      .pipe(finalize(() => this.documentsLoading.set(false)))
      .subscribe({
        next: (documents) => {
          this.documents.set(documents);
          const availableIds = new Set(documents.map((document) => document.id));
          this.batchSelectedDocumentIds.update((ids) =>
            ids.filter((documentId) => availableIds.has(documentId)),
          );
        },
        error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
      });
  }

  loadDocumentDetail(documentId: string): void {
    const requestId = ++this.documentDetailRequest;
    this.activeDocumentId = documentId;
    ++this.documentSubjectsRequest;
    this.selectedDocument.set(null);
    this.selectedDocumentDecisions.set([]);
    this.documentSubjectsLoading.set(true);
    this.documentError.set(null);

    forkJoin({
      document: this.documentsApi.getDocument(documentId),
      decisions: this.subjectsApi.listDocumentDecisions(documentId),
    }).pipe(
      finalize(() => {
        if (
          this.activeDocumentId === documentId &&
          this.documentDetailRequest === requestId
        ) {
          this.documentSubjectsLoading.set(false);
        }
      }),
    ).subscribe({
      next: ({ document, decisions }) => {
        if (
          this.activeDocumentId !== documentId ||
          this.documentDetailRequest !== requestId
        ) {
          return;
        }
        this.selectedDocument.set(document);
        this.selectedDocumentDecisions.set(decisions);
      },
      error: (error: unknown) => {
        if (
          this.activeDocumentId === documentId &&
          this.documentDetailRequest === requestId
        ) {
          this.documentError.set(toApiErrorMessage(error));
        }
      },
    });
  }

  uploadDocuments(request: DocumentUploadRequest): void {
    this.documentUploading.set(true);
    this.documentError.set(null);

    this.documentsApi
      .uploadDocuments(request)
      .pipe(finalize(() => this.documentUploading.set(false)))
      .subscribe({
        next: (result) => {
          for (const upload of result.accepted) {
            this.trackJob(upload.jobId);
          }
          const latestAccepted = result.accepted.at(-1);
          if (latestAccepted) {
            this.activeDocumentId = latestAccepted.document.id;
            ++this.documentDetailRequest;
            this.selectedDocument.set(latestAccepted.document);
            this.loadDocumentSubjects(latestAccepted.document.id);
          }
          this.loadDocuments();
          if (result.rejected.length > 0) {
            this.documentError.set(
              result.rejected
                .map((item) => `${item.filename}: ${item.detail}`)
                .join('\n'),
            );
          }
        },
        error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
      });
  }

  setDocumentSubjectDecision(request: DocumentSubjectChangeRequest): void {
    const document = this.selectedDocument();
    if (!document) {
      return;
    }
    this.subjectMutationBusy.set(true);
    this.documentError.set(null);
    this.subjectsApi
      .setDocumentDecision(
        document.id,
        request.subjectId,
        request.state,
        request.expectedRevision,
      )
      .pipe(finalize(() => this.subjectMutationBusy.set(false)))
      .subscribe({
        next: () => {
          if (this.activeDocumentId === document.id) {
            this.loadDocumentSubjects(document.id);
          }
        },
        error: (error: unknown) => {
          if (this.activeDocumentId === document.id) {
            this.documentError.set(toApiErrorMessage(error));
            this.loadDocumentSubjects(document.id);
          }
        },
      });
  }

  reviewDocumentSuggestion(request: DocumentSuggestionReviewRequest): void {
    const document = this.selectedDocument();
    if (!document) {
      return;
    }
    this.subjectMutationBusy.set(true);
    this.documentError.set(null);
    this.subjectsApi
      .reviewSuggestion(
        document.id,
        request.subjectId,
        request.decision,
        request.expectedRevision,
      )
      .pipe(finalize(() => this.subjectMutationBusy.set(false)))
      .subscribe({
        next: () => {
          if (this.activeDocumentId === document.id) {
            this.loadDocumentSubjects(document.id);
          }
        },
        error: (error: unknown) => {
          if (this.activeDocumentId === document.id) {
            this.documentError.set(toApiErrorMessage(error));
            this.loadDocumentSubjects(document.id);
          }
        },
      });
  }

  reclassifySelectedDocument(): void {
    const document = this.selectedDocument();
    if (!document || this.selectedClassificationBusy()) {
      return;
    }
    this.classificationSubmitting.set(true);
    this.documentError.set(null);
    this.documentsApi.reclassifyDocumentSubjects(document.id).pipe(
      finalize(() => this.classificationSubmitting.set(false)),
    ).subscribe({
      next: (result) => {
        const selectedJob = result.jobs.find((job) => job.document_id === document.id);
        if (selectedJob && this.activeDocumentId === document.id) {
          const previous = document.subject_classification;
          this.selectedDocument.set({
            ...document,
            subject_classification: {
              status: 'queued',
              job_id: selectedJob.job_id,
              document_version_id: selectedJob.document_version_id,
              policy_version: previous?.policy_version ?? null,
              classifier_version: previous?.classifier_version ?? null,
              assigned_count: previous?.assigned_count ?? null,
              suggested_count: previous?.suggested_count ?? null,
              review_required_count: previous?.review_required_count ?? null,
              error_message: null,
            },
          });
        }
        for (const queued of result.jobs) {
          this.trackJob(queued.job_id);
        }
        if (result.jobs.length === 0 && this.activeDocumentId === document.id) {
          this.loadDocumentDetail(document.id);
        }
      },
      error: (error: unknown) => {
        if (this.activeDocumentId === document.id) {
          this.documentError.set(toApiErrorMessage(error));
        }
      },
    });
  }

  private loadSubjects(): void {
    this.subjectsApi.listSubjects().subscribe({
      next: (subjects) => this.subjects.set(subjects),
      error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
    });
  }

  private loadDocumentSubjects(documentId: string): void {
    if (this.activeDocumentId !== documentId) {
      return;
    }
    const requestId = ++this.documentSubjectsRequest;
    this.selectedDocumentDecisions.set([]);
    this.documentSubjectsLoading.set(true);
    this.subjectsApi.listDocumentDecisions(documentId).pipe(
      finalize(() => {
        if (
          this.activeDocumentId === documentId &&
          this.documentSubjectsRequest === requestId
        ) {
          this.documentSubjectsLoading.set(false);
        }
      }),
    ).subscribe({
      next: (decisions) => {
        if (
          this.activeDocumentId === documentId &&
          this.documentSubjectsRequest === requestId
        ) {
          this.selectedDocumentDecisions.set(decisions);
        }
      },
      error: (error: unknown) => {
        if (
          this.activeDocumentId === documentId &&
          this.documentSubjectsRequest === requestId
        ) {
          this.documentError.set(toApiErrorMessage(error));
        }
      },
    });
  }

  updateBatchDocumentSelection(change: DocumentBatchSelectionChange): void {
    this.batchSelectedDocumentIds.update((ids) => {
      const selected = new Set(ids);
      if (change.selected) {
        selected.add(change.documentId);
      } else {
        selected.delete(change.documentId);
      }
      return [...selected];
    });
  }

  deleteSelectedDocuments(): void {
    const documentIds = this.batchSelectedDocumentIds();
    if (documentIds.length === 0) {
      return;
    }
    const confirmed = window.confirm(
      `Remove every version of ${documentIds.length} selected document(s)? ` +
        'Each version is deleted explicitly, and empty document records are removed. This cannot be undone.',
    );
    if (!confirmed) {
      return;
    }

    this.deletionSubmitting.set(true);
    this.documentError.set(null);
    forkJoin(documentIds.map((documentId) => this.documentsApi.getDocument(documentId)))
      .pipe(
        switchMap((documents) =>
          this.documentsApi.deleteDocumentVersions(
            documents.flatMap((document) =>
              document.versions.map((version) => ({
                documentId: document.id,
                versionId: version.id,
              })),
            ),
          ),
        ),
        finalize(() => this.deletionSubmitting.set(false)),
      )
      .subscribe({
        next: (queued) => {
          this.batchSelectedDocumentIds.set([]);
          this.trackJob(queued.job_id);
        },
        error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
      });
  }

  deleteDocumentVersions(request: DeleteDocumentVersionsRequest): void {
    const versionLabels = request.versions
      .map((version) => `v${version.version_number}`)
      .join(', ');
    const finalVersionWarning =
      request.versions.length === request.document.versions.length
        ? ' This also removes the document record because no versions will remain.'
        : '';
    const confirmed = window.confirm(
      `Remove ${versionLabels} from “${request.document.title}”?${finalVersionWarning} This cannot be undone.`,
    );
    if (!confirmed) {
      return;
    }

    this.deletionSubmitting.set(true);
    this.documentError.set(null);
    const operation =
      request.versions.length === 1
        ? this.documentsApi.deleteDocumentVersion(
            request.document.id,
            request.versions[0].id,
          )
        : this.documentsApi.deleteDocumentVersions(
            request.versions.map((version) => ({
              documentId: request.document.id,
              versionId: version.id,
            })),
          );

    operation.pipe(finalize(() => this.deletionSubmitting.set(false))).subscribe({
      next: (queued) => this.trackJob(queued.job_id),
      error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
    });
  }

  private resumeActiveDocumentJobs(): void {
    this.jobsApi.listJobs().subscribe({
      next: (jobs) => {
        for (const job of jobs) {
          if (isRunningJob(job) && DOCUMENT_JOB_TYPES.has(job.job_type)) {
            this.trackJob(job.id, job);
          }
        }
      },
      error: () => {
        // Job recovery is best-effort; normal document reads remain available.
      },
    });
  }

  private trackJob(jobId: string, initialJob: BackgroundJob | null = null): void {
    if (initialJob) {
      this.upsertTrackedJob(initialJob);
    }
    if (this.jobPolling.has(jobId)) {
      return;
    }

    const subscription = timer(0, 1000)
      .pipe(
        switchMap(() => this.jobsApi.getJob(jobId)),
        takeWhile((job) => !isTerminalJob(job), true),
      )
      .subscribe({
        next: (job) => {
          this.upsertTrackedJob(job);
          if (isTerminalJob(job)) {
            this.jobPolling.delete(job.id);
            this.handleTerminalJob(job);
          }
        },
        error: (error: unknown) => {
          this.jobPolling.delete(jobId);
          this.documentError.set(
            `A background job could not be refreshed: ${toApiErrorMessage(error)}`,
          );
        },
      });
    this.jobPolling.set(jobId, subscription);
  }

  private upsertTrackedJob(job: BackgroundJob): void {
    this.trackedJobs.update((jobs) => {
      const updated = [job, ...jobs.filter((item) => item.id !== job.id)];
      return updated.slice(0, 12);
    });
  }

  private handleTerminalJob(job: BackgroundJob): void {
    this.loadDocuments();
    const selected = this.selectedDocument();
    const affectsSelected = selected !== null && documentIdsForJob(job).has(selected.id);
    if (job.status === 'failed') {
      if (selected && affectsSelected) {
        this.loadDocumentDetail(selected.id);
      }
      this.documentError.set(job.error_message || 'The background operation failed.');
      return;
    }
    if (job.status === 'cancelled') {
      if (selected && affectsSelected) {
        this.loadDocumentDetail(selected.id);
      }
      this.documentError.set('The background operation was cancelled.');
      return;
    }

    if (!selected) {
      return;
    }
    if (!affectsSelected) {
      return;
    }
    const deletedDocumentIds = stringSet(job.result['deleted_document_ids']);
    if (deletedDocumentIds.has(selected.id)) {
      this.activeDocumentId = null;
      ++this.documentDetailRequest;
      ++this.documentSubjectsRequest;
      this.selectedDocument.set(null);
      this.selectedDocumentDecisions.set([]);
      this.documentSubjectsLoading.set(false);
      return;
    }
    this.loadDocumentDetail(selected.id);
  }
}

const DOCUMENT_JOB_TYPES = new Set([
  'ingest_document',
  'rebuild_document_index',
  'contextualize_document',
  'delete_document_versions',
  'classify_document_subjects',
]);

function isRunningJob(job: BackgroundJob): boolean {
  return job.status === 'queued' || job.status === 'running';
}

function isTerminalJob(job: BackgroundJob): boolean {
  return job.status === 'succeeded' || job.status === 'failed' || job.status === 'cancelled';
}

function documentIdsForJob(job: BackgroundJob): Set<string> {
  const ids = new Set<string>();
  const documentId = job.payload['document_id'];
  if (typeof documentId === 'string') {
    ids.add(documentId);
  }
  const targets = job.payload['targets'];
  if (Array.isArray(targets)) {
    for (const target of targets) {
      if (isRecord(target) && typeof target['document_id'] === 'string') {
        ids.add(target['document_id']);
      }
    }
  }
  return ids;
}

function stringSet(value: unknown): Set<string> {
  return new Set(Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
