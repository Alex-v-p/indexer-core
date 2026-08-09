import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
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
  DocumentGroupChangeRequest,
  DocumentOrganizationPanelComponent,
  DocumentTypesChangeRequest,
} from '../../components/document-organization-panel/document-organization-panel.component';
import { BackgroundJobsApiService } from '../../data-access/background-jobs-api.service';
import { DocumentsApiService } from '../../data-access/documents-api.service';
import { BackgroundJob } from '../../models/background-job.models';
import { DocumentDetail, DocumentSummary, DocumentUploadRequest } from '../../models/document.models';
import { DocumentOrganizationApiService } from '../../../organization/data-access/document-organization-api.service';
import {
  ContentGroup,
  DocumentOrganization,
  DocumentType,
} from '../../../organization/models/document-organization.models';

@Component({
  selector: 'app-documents-page',
  standalone: true,
  imports: [
    DocumentUploadComponent,
    DocumentListComponent,
    DocumentMetadataPanelComponent,
    DocumentJobProgressComponent,
    DocumentOrganizationPanelComponent,
  ],
  templateUrl: './documents-page.component.html',
})
export class DocumentsPageComponent implements OnInit, OnDestroy {
  private readonly documentsApi = inject(DocumentsApiService);
  private readonly jobsApi = inject(BackgroundJobsApiService);
  private readonly organizationApi = inject(DocumentOrganizationApiService);
  private readonly jobPolling = new Map<string, Subscription>();
  private activeDocumentId: string | null = null;
  private documentDetailRequest = 0;
  private documentOrganizationRequest = 0;

  readonly documents = signal<DocumentSummary[]>([]);
  readonly selectedDocument = signal<DocumentDetail | null>(null);
  readonly contentGroups = signal<ContentGroup[]>([]);
  readonly documentTypes = signal<DocumentType[]>([]);
  readonly selectedDocumentOrganization = signal<DocumentOrganization | null>(null);
  readonly batchSelectedDocumentIds = signal<string[]>([]);
  readonly trackedJobs = signal<BackgroundJob[]>([]);
  readonly documentsLoading = signal(false);
  readonly documentUploading = signal(false);
  readonly deletionSubmitting = signal(false);
  readonly organizationMutationBusy = signal(false);
  readonly classificationSubmitting = signal(false);
  readonly documentOrganizationLoading = signal(false);
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
    const status = this.selectedDocumentOrganization()?.status;
    const rawJobId = status?.['job_id'];
    const jobId = typeof rawJobId === 'string' ? rawJobId : null;
    const tracked = jobId
      ? this.trackedJobs().find((job) => job.id === jobId)
      : undefined;
    if (tracked) {
      return isRunningJob(tracked);
    }
    return status?.['status'] === 'queued' || status?.['status'] === 'running';
  });

  ngOnInit(): void {
    this.loadDocuments();
    this.loadOrganizationCatalogues();
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
    ++this.documentOrganizationRequest;
    this.selectedDocument.set(null);
    this.selectedDocumentOrganization.set(null);
    this.documentOrganizationLoading.set(true);
    this.documentError.set(null);

    forkJoin({
      document: this.documentsApi.getDocument(documentId),
      organization: this.organizationApi.getDocumentOrganization(documentId),
    }).pipe(
      finalize(() => {
        if (
          this.activeDocumentId === documentId &&
          this.documentDetailRequest === requestId
        ) {
          this.documentOrganizationLoading.set(false);
        }
      }),
    ).subscribe({
      next: ({ document, organization }) => {
        if (
          this.activeDocumentId !== documentId ||
          this.documentDetailRequest !== requestId
        ) {
          return;
        }
        this.selectedDocument.set(document);
        this.acceptDocumentOrganization(organization);
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
            this.loadDocumentOrganization(latestAccepted.document.id);
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

  setDocumentContentGroup(request: DocumentGroupChangeRequest): void {
    const document = this.selectedDocument();
    if (!document) {
      return;
    }
    this.organizationMutationBusy.set(true);
    this.documentError.set(null);
    this.organizationApi
      .setDocumentContentGroup(
        document.id,
        request.contentGroupId,
        request.expectedRevision,
      )
      .pipe(finalize(() => this.organizationMutationBusy.set(false)))
      .subscribe({
        next: () => {
          if (this.activeDocumentId === document.id) {
            this.loadDocumentOrganization(document.id);
          }
        },
        error: (error: unknown) => {
          if (this.activeDocumentId === document.id) {
            this.handleOrganizationMutationError(error);
            this.loadDocumentOrganization(document.id);
          }
        },
      });
  }

  setDocumentTypes(request: DocumentTypesChangeRequest): void {
    const document = this.selectedDocument();
    if (!document) {
      return;
    }
    this.organizationMutationBusy.set(true);
    this.documentError.set(null);
    this.organizationApi
      .replaceDocumentTypes(document.id, request.decisions)
      .pipe(finalize(() => this.organizationMutationBusy.set(false)))
      .subscribe({
        next: () => {
          if (this.activeDocumentId === document.id) {
            this.loadDocumentOrganization(document.id);
          }
        },
        error: (error: unknown) => {
          if (this.activeDocumentId === document.id) {
            this.handleOrganizationMutationError(error);
            this.loadDocumentOrganization(document.id);
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
    this.organizationApi.requeueDocument(document.id).pipe(
      finalize(() => this.classificationSubmitting.set(false)),
    ).subscribe({
      next: (result) => {
        const selectedJobId = result.job_ids[0];
        if (selectedJobId && this.activeDocumentId === document.id) {
          const current = this.selectedDocumentOrganization();
          if (current) {
            this.selectedDocumentOrganization.set({
              ...current,
              status: { ...(current.status ?? {}), status: 'queued', job_id: selectedJobId },
            });
          }
        }
        for (const jobId of result.job_ids) {
          this.trackJob(jobId);
        }
        if (result.job_ids.length === 0 && this.activeDocumentId === document.id) {
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

  private loadOrganizationCatalogues(): void {
    forkJoin({
      groups: this.organizationApi.listContentGroups(),
      types: this.organizationApi.listDocumentTypes(),
    }).subscribe({
      next: ({ groups, types }) => {
        this.contentGroups.set(groups);
        this.documentTypes.set(types);
      },
      error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
    });
  }

  private loadDocumentOrganization(documentId: string): void {
    if (this.activeDocumentId !== documentId) {
      return;
    }
    const requestId = ++this.documentOrganizationRequest;
    this.selectedDocumentOrganization.set(null);
    this.documentOrganizationLoading.set(true);
    this.organizationApi.getDocumentOrganization(documentId).pipe(
      finalize(() => {
        if (
          this.activeDocumentId === documentId &&
          this.documentOrganizationRequest === requestId
        ) {
          this.documentOrganizationLoading.set(false);
        }
      }),
    ).subscribe({
      next: (organization) => {
        if (
          this.activeDocumentId === documentId &&
          this.documentOrganizationRequest === requestId
        ) {
          this.acceptDocumentOrganization(organization);
        }
      },
      error: (error: unknown) => {
        if (
          this.activeDocumentId === documentId &&
          this.documentOrganizationRequest === requestId
        ) {
          this.documentError.set(toApiErrorMessage(error));
        }
      },
    });
  }

  private handleOrganizationMutationError(error: unknown): void {
    this.documentError.set(
      error instanceof HttpErrorResponse && error.status === 409
        ? `This document changed before your update was saved. The latest organization has been loaded. ${toApiErrorMessage(error)}`
        : toApiErrorMessage(error),
    );
  }

  private acceptDocumentOrganization(organization: DocumentOrganization): void {
    this.selectedDocumentOrganization.set(organization);
    const status = organization.status?.['status'];
    const jobId = organization.status?.['job_id'];
    if (
      (status === 'queued' || status === 'running') &&
      typeof jobId === 'string' &&
      jobId.length > 0
    ) {
      this.trackJob(jobId);
    }
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
      ++this.documentOrganizationRequest;
      this.selectedDocument.set(null);
      this.selectedDocumentOrganization.set(null);
      this.documentOrganizationLoading.set(false);
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
  'classify_document_organization',
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
