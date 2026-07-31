import { HttpClient, HttpResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { QueuedDocumentOperation } from '../models/background-job.models';
import {
  BatchQueuedDocumentUpload,
  DocumentDetail,
  DocumentSummary,
  DocumentUploadRequest,
  DocumentVersionDeletionTarget,
  QueuedDocumentUpload,
} from '../models/document.models';

interface BatchUploadResponse {
  accepted: Array<{
    filename: string;
    document: DocumentDetail;
    job_id: string;
  }>;
  rejected: Array<{
    filename: string;
    detail: string;
  }>;
}

@Injectable({ providedIn: 'root' })
export class DocumentsApiService {
  private readonly http = inject(HttpClient);

  listDocuments(): Observable<DocumentSummary[]> {
    return this.http.get<DocumentSummary[]>('/documents');
  }

  getDocument(documentId: string): Observable<DocumentDetail> {
    return this.http.get<DocumentDetail>(`/documents/${documentId}`);
  }

  uploadDocuments(request: DocumentUploadRequest): Observable<BatchQueuedDocumentUpload> {
    if (request.files.length === 1) {
      return this.uploadSingleDocument(request, request.files[0]).pipe(
        map((accepted) => ({ accepted: [accepted], rejected: [] })),
      );
    }

    const formData = this.toFormData(request);
    for (const file of request.files) {
      formData.append('files', file);
    }
    if (request.versionOfDocumentId) {
      formData.append('version_of_document_id', request.versionOfDocumentId);
    }

    return this.http.post<BatchUploadResponse>('/documents/batch', formData).pipe(
      map((response) => ({
        accepted: response.accepted.map((item) => ({
          filename: item.filename,
          document: item.document,
          jobId: item.job_id,
        })),
        rejected: response.rejected,
      })),
    );
  }

  deleteDocumentVersion(
    documentId: string,
    versionId: string,
  ): Observable<QueuedDocumentOperation> {
    return this.http.delete<QueuedDocumentOperation>(
      `/documents/${documentId}/versions/${versionId}`,
    );
  }

  deleteDocumentVersions(
    targets: DocumentVersionDeletionTarget[],
  ): Observable<QueuedDocumentOperation> {
    return this.http.post<QueuedDocumentOperation>('/documents/versions/batch-delete', {
      targets: targets.map((target) => ({
        document_id: target.documentId,
        document_version_id: target.versionId,
      })),
    });
  }

  private uploadSingleDocument(
    request: DocumentUploadRequest,
    file: File,
  ): Observable<QueuedDocumentUpload> {
    const formData = this.toFormData(request);
    formData.append('file', file);

    const response = request.versionOfDocumentId
      ? this.http.post<DocumentDetail>(
          `/documents/${request.versionOfDocumentId}/versions`,
          formData,
          { observe: 'response' },
        )
      : this.uploadNewDocument(formData, request.detectExistingVersions);

    return response.pipe(
      map((httpResponse) => ({
        ...this.toQueuedUpload(httpResponse),
        filename: file.name,
      })),
    );
  }

  private toFormData(request: DocumentUploadRequest): FormData {
    const formData = new FormData();
    if (request.title) {
      formData.append('title', request.title);
    }
    if (request.publishedAt) {
      formData.append('published_at', request.publishedAt);
    }
    formData.append('detect_existing_versions', String(request.detectExistingVersions));
    return formData;
  }

  private uploadNewDocument(
    formData: FormData,
    detectExistingVersions: boolean,
  ): Observable<HttpResponse<DocumentDetail>> {
    formData.set('detect_existing_versions', String(detectExistingVersions));
    return this.http.post<DocumentDetail>('/documents', formData, { observe: 'response' });
  }

  private toQueuedUpload(response: HttpResponse<DocumentDetail>): Omit<QueuedDocumentUpload, 'filename'> {
    if (response.body === null) {
      throw new Error('The upload response did not include the queued document.');
    }
    const jobId = response.headers.get('X-Background-Job-ID');
    if (!jobId) {
      throw new Error('The upload response did not include a background job ID.');
    }
    return { document: response.body, jobId };
  }
}
