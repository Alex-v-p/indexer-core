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
import { BackgroundJobsApiService } from '../../data-access/background-jobs-api.service';
import { DocumentsApiService } from '../../data-access/documents-api.service';
import { BackgroundJob } from '../../models/background-job.models';
import { DocumentDetail, DocumentSummary, DocumentUploadRequest } from '../../models/document.models';

@Component({
  selector: 'app-documents-page',
  standalone: true,
  imports: [
    DocumentUploadComponent,
    DocumentListComponent,
    DocumentMetadataPanelComponent,
    DocumentJobProgressComponent,
  ],
  templateUrl: './documents-page.component.html',
})
export class DocumentsPageComponent implements OnInit, OnDestroy {
  private readonly documentsApi = inject(DocumentsApiService);
  private readonly jobsApi = inject(BackgroundJobsApiService);
  private readonly jobPolling = new Map<string, Subscription>();

  readonly documents = signal<DocumentSummary[]>([]);
  readonly selectedDocument = signal<DocumentDetail | null>(null);
  readonly batchSelectedDocumentIds = signal<string[]>([]);
  readonly trackedJobs = signal<BackgroundJob[]>([]);
  readonly documentsLoading = signal(false);
  readonly documentUploading = signal(false);
  readonly deletionSubmitting = signal(false);
  readonly documentError = signal<string | null>(null);
  readonly versionDeletionBusy = computed(
    () =>
      this.deletionSubmitting() ||
      this.trackedJobs().some(
        (job) => job.job_type === 'delete_document_versions' && isRunningJob(job),
      ),
  );

  ngOnInit(): void {
    this.loadDocuments();
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
    this.documentError.set(null);

    this.documentsApi.getDocument(documentId).subscribe({
      next: (document) => this.selectedDocument.set(document),
      error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
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
            this.selectedDocument.set(latestAccepted.document);
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
    if (job.status === 'failed') {
      this.documentError.set(job.error_message || 'The background operation failed.');
      return;
    }
    if (job.status === 'cancelled') {
      this.documentError.set('The background operation was cancelled.');
      return;
    }

    const selected = this.selectedDocument();
    if (!selected) {
      return;
    }
    const affectedDocumentIds = documentIdsForJob(job);
    if (!affectedDocumentIds.has(selected.id)) {
      return;
    }
    const deletedDocumentIds = stringSet(job.result['deleted_document_ids']);
    if (deletedDocumentIds.has(selected.id)) {
      this.selectedDocument.set(null);
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
