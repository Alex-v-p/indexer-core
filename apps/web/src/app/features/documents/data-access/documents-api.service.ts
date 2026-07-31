import { HttpClient, HttpResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { QueuedDocumentOperation } from '../models/background-job.models';
import {
  DocumentDetail,
  DocumentSummary,
  DocumentUploadRequest,
  QueuedDocumentUpload,
} from '../models/document.models';

@Injectable({ providedIn: 'root' })
export class DocumentsApiService {
  private readonly http = inject(HttpClient);

  listDocuments(): Observable<DocumentSummary[]> {
    return this.http.get<DocumentSummary[]>('/documents');
  }

  getDocument(documentId: string): Observable<DocumentDetail> {
    return this.http.get<DocumentDetail>(`/documents/${documentId}`);
  }

  uploadDocument(request: DocumentUploadRequest): Observable<QueuedDocumentUpload> {
    const formData = new FormData();
    formData.append('file', request.file);

    if (request.title) {
      formData.append('title', request.title);
    }
    if (request.publishedAt) {
      formData.append('published_at', request.publishedAt);
    }

    const response = request.versionOfDocumentId
      ? this.http.post<DocumentDetail>(
          `/documents/${request.versionOfDocumentId}/versions`,
          formData,
          { observe: 'response' },
        )
      : this.uploadNewDocument(formData, request.detectExistingVersions);

    return response.pipe(map((httpResponse) => this.toQueuedUpload(httpResponse)));
  }

  deleteDocument(documentId: string): Observable<QueuedDocumentOperation> {
    return this.http.delete<QueuedDocumentOperation>(`/documents/${documentId}`);
  }

  private uploadNewDocument(
    formData: FormData,
    detectExistingVersions: boolean,
  ): Observable<HttpResponse<DocumentDetail>> {
    formData.append('detect_existing_versions', String(detectExistingVersions));
    return this.http.post<DocumentDetail>('/documents', formData, { observe: 'response' });
  }

  private toQueuedUpload(response: HttpResponse<DocumentDetail>): QueuedDocumentUpload {
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
