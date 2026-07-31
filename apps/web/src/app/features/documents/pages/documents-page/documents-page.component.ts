import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { Subscription, finalize, switchMap, takeWhile, timer } from 'rxjs';

import { toApiErrorMessage } from '../../../../core/http/api-error';
import { DocumentJobProgressComponent } from '../../components/document-job-progress/document-job-progress.component';
import { DocumentListComponent } from '../../components/document-list/document-list.component';
import { DocumentMetadataPanelComponent } from '../../components/document-metadata-panel/document-metadata-panel.component';
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
  private jobPolling: Subscription | null = null;

  readonly documents = signal<DocumentSummary[]>([]);
  readonly selectedDocument = signal<DocumentDetail | null>(null);
  readonly activeJob = signal<BackgroundJob | null>(null);
  readonly documentsLoading = signal(false);
  readonly documentUploading = signal(false);
  readonly documentDeleting = signal(false);
  readonly documentError = signal<string | null>(null);
  readonly activeJobDocumentId = signal<string | null>(null);
  readonly activeJobRunning = computed(() => {
    const status = this.activeJob()?.status;
    return status === 'queued' || status === 'running';
  });
  readonly selectedDocumentDeleting = computed(() => {
    const selected = this.selectedDocument();
    const job = this.activeJob();
    return Boolean(
      selected &&
        job?.job_type === 'delete_document' &&
        this.activeJobDocumentId() === selected.id &&
        this.activeJobRunning(),
    );
  });

  ngOnInit(): void {
    this.loadDocuments();
    this.resumeActiveDocumentJob();
  }

  ngOnDestroy(): void {
    this.jobPolling?.unsubscribe();
  }

  loadDocuments(): void {
    this.documentsLoading.set(true);
    this.documentError.set(null);

    this.documentsApi
      .listDocuments()
      .pipe(finalize(() => this.documentsLoading.set(false)))
      .subscribe({
        next: (documents) => this.documents.set(documents),
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

  uploadDocument(request: DocumentUploadRequest): void {
    this.documentUploading.set(true);
    this.documentError.set(null);

    this.documentsApi
      .uploadDocument(request)
      .pipe(finalize(() => this.documentUploading.set(false)))
      .subscribe({
        next: ({ document, jobId }) => {
          this.selectedDocument.set(document);
          this.loadDocuments();
          this.trackJob(jobId, document.id);
        },
        error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
      });
  }

  deleteDocument(document: DocumentDetail): void {
    const confirmed = window.confirm(
      `Remove “${document.title}” and every stored version? This cannot be undone.`,
    );
    if (!confirmed) {
      return;
    }

    this.documentDeleting.set(true);
    this.documentError.set(null);
    this.documentsApi.deleteDocument(document.id).subscribe({
      next: (operation) => this.trackJob(operation.job_id, document.id),
      error: (error: unknown) => {
        this.documentDeleting.set(false);
        this.documentError.set(toApiErrorMessage(error));
      },
    });
  }

  private resumeActiveDocumentJob(): void {
    this.jobsApi.listJobs().subscribe({
      next: (jobs) => {
        if (this.activeJob() !== null) {
          return;
        }
        const active = jobs.find(
          (job) =>
            (job.status === 'queued' || job.status === 'running') &&
            DOCUMENT_JOB_TYPES.has(job.job_type) &&
            typeof job.payload['document_id'] === 'string',
        );
        const documentId = active?.payload['document_id'];
        if (active && typeof documentId === 'string') {
          this.trackJob(active.id, documentId, active);
        }
      },
      error: () => {
        // Job recovery is best-effort; normal document reads remain available.
      },
    });
  }

  private trackJob(
    jobId: string,
    documentId: string,
    initialJob: BackgroundJob | null = null,
  ): void {
    this.jobPolling?.unsubscribe();
    this.activeJob.set(initialJob);
    this.activeJobDocumentId.set(documentId);

    this.jobPolling = timer(0, 1000)
      .pipe(
        switchMap(() => this.jobsApi.getJob(jobId)),
        takeWhile((job) => !isTerminalJob(job), true),
      )
      .subscribe({
        next: (job) => {
          this.activeJob.set(job);
          if (isTerminalJob(job)) {
            this.handleTerminalJob(job, documentId);
          }
        },
        error: (error: unknown) => {
          this.documentDeleting.set(false);
          this.documentError.set(
            `The background job could not be refreshed: ${toApiErrorMessage(error)}`,
          );
        },
      });
  }

  private handleTerminalJob(job: BackgroundJob, documentId: string): void {
    this.loadDocuments();
    if (job.job_type === 'delete_document') {
      this.documentDeleting.set(false);
    }
    if (job.status === 'failed') {
      this.documentError.set(job.error_message || 'The background operation failed.');
      return;
    }
    if (job.status === 'cancelled') {
      this.documentError.set('The background operation was cancelled.');
      return;
    }
    if (job.job_type === 'delete_document') {
      if (this.selectedDocument()?.id === documentId) {
        this.selectedDocument.set(null);
      }
      return;
    }
    this.loadDocumentDetail(documentId);
  }
}

const DOCUMENT_JOB_TYPES = new Set([
  'ingest_document',
  'rebuild_document_index',
  'contextualize_document',
  'delete_document',
]);

function isTerminalJob(job: BackgroundJob): boolean {
  return job.status === 'succeeded' || job.status === 'failed' || job.status === 'cancelled';
}
