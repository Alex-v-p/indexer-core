import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { DocumentDetail, DocumentSummary } from '../models/document.models';

@Injectable({ providedIn: 'root' })
export class DocumentsApiService {
  private readonly http = inject(HttpClient);

  listDocuments(): Observable<DocumentSummary[]> {
    return this.http.get<DocumentSummary[]>('/documents');
  }

  getDocument(documentId: string): Observable<DocumentDetail> {
    return this.http.get<DocumentDetail>(`/documents/${documentId}`);
  }

  uploadDocument(file: File, title?: string): Observable<DocumentDetail> {
    const formData = new FormData();
    formData.append('file', file);

    const normalizedTitle = title?.trim();
    if (normalizedTitle) {
      formData.append('title', normalizedTitle);
    }

    return this.http.post<DocumentDetail>('/documents', formData);
  }
}
