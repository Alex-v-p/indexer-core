import { Component, OnInit, inject, signal } from '@angular/core';
import { finalize } from 'rxjs';

import { toApiErrorMessage } from '../../../../core/http/api-error';
import { DocumentListComponent } from '../../components/document-list/document-list.component';
import { DocumentMetadataPanelComponent } from '../../components/document-metadata-panel/document-metadata-panel.component';
import {
  DocumentUploadComponent,
  DocumentUploadRequest,
} from '../../components/document-upload/document-upload.component';
import { DocumentsApiService } from '../../data-access/documents-api.service';
import { DocumentDetail, DocumentSummary } from '../../models/document.models';

@Component({
  selector: 'app-documents-page',
  standalone: true,
  imports: [DocumentUploadComponent, DocumentListComponent, DocumentMetadataPanelComponent],
  templateUrl: './documents-page.component.html',
})
export class DocumentsPageComponent implements OnInit {
  private readonly documentsApi = inject(DocumentsApiService);

  readonly documents = signal<DocumentSummary[]>([]);
  readonly selectedDocument = signal<DocumentDetail | null>(null);
  readonly documentsLoading = signal(false);
  readonly documentUploading = signal(false);
  readonly documentError = signal<string | null>(null);

  ngOnInit(): void {
    this.loadDocuments();
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
      .uploadDocument(request.file, request.title)
      .pipe(finalize(() => this.documentUploading.set(false)))
      .subscribe({
        next: (document) => {
          this.selectedDocument.set(document);
          this.loadDocuments();
        },
        error: (error: unknown) => this.documentError.set(toApiErrorMessage(error)),
      });
  }
}
