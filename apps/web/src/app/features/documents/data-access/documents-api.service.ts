import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { DocumentDetail, DocumentSummary, DocumentUploadRequest } from '../models/document.models';

@Injectable({ providedIn: 'root' })
export class DocumentsApiService {
  private readonly http = inject(HttpClient);

  listDocuments(): Observable<DocumentSummary[]> {
    return this.http.get<DocumentSummary[]>('/documents');
  }

  getDocument(documentId: string): Observable<DocumentDetail> {
    return this.http.get<DocumentDetail>(`/documents/${documentId}`);
  }

  uploadDocument(request: DocumentUploadRequest): Observable<DocumentDetail> {
    const formData = new FormData();
    formData.append('file', request.file);

    if (request.title) {
      formData.append('title', request.title);
    }
    if (request.publishedAt) {
      formData.append('published_at', request.publishedAt);
    }

    if (request.versionOfDocumentId) {
      return this.http.post<DocumentDetail>(
        `/documents/${request.versionOfDocumentId}/versions`,
        formData,
      );
    }

    formData.append('detect_existing_versions', String(request.detectExistingVersions));
    return this.http.post<DocumentDetail>('/documents', formData);
  }
}
