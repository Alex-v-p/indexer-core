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
  template: `
    <section class="page-stack" aria-label="Documents workspace">
      <header class="page-heading">
        <div>
          <p class="page-heading__eyebrow">Knowledge base</p>
          <h1>Documents</h1>
          <p>
            Upload source files and inspect the metadata that Indexer Core stores for retrieval,
            chunking, and citation support.
          </p>
        </div>
        <button class="ghost-button" type="button" (click)="loadDocuments()" [disabled]="documentsLoading()">
          {{ documentsLoading() ? 'Refreshing…' : 'Refresh documents' }}
        </button>
      </header>

      <div class="documents-layout">
        <section class="panel">
          <div class="panel__header">
            <div>
              <p class="panel__eyebrow">Upload</p>
              <h2>Add a document</h2>
            </div>
          </div>

          <app-document-upload
            [uploading]="documentUploading()"
            [error]="documentError()"
            (uploadRequested)="uploadDocument($event)"
          />
        </section>

        <section class="panel documents-panel">
          <div class="panel__header">
            <div>
              <p class="panel__eyebrow">Library</p>
              <h2>Indexed documents</h2>
            </div>
          </div>

          <app-document-list
            [documents]="documents()"
            [loading]="documentsLoading()"
            [selectedDocumentId]="selectedDocument()?.id ?? null"
            (selectDocument)="loadDocumentDetail($event)"
          />
        </section>

        <section class="panel metadata-panel">
          <div class="panel__header">
            <div>
              <p class="panel__eyebrow">Metadata</p>
              <h2>Selected document</h2>
            </div>
          </div>

          @if (selectedDocument()) {
            <app-document-metadata-panel [document]="selectedDocument()" />
          } @else {
            <p class="empty-state">Select a document to inspect versions, checksum data, and chunk index metadata.</p>
          }
        </section>
      </div>
    </section>
  `,
  styles: [
    `
      .page-stack {
        display: grid;
        gap: 24px;
      }

      .page-heading {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 24px;
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        background: var(--surface);
        box-shadow: var(--shadow);
        padding: 28px;
      }

      .page-heading__eyebrow,
      .panel__eyebrow {
        margin: 0 0 6px;
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      h1,
      h2 {
        margin: 0;
        letter-spacing: -0.04em;
      }

      h1 {
        font-size: clamp(2.25rem, 5vw, 4.25rem);
        line-height: 0.95;
      }

      h2 {
        font-size: 1.35rem;
      }

      .page-heading p:not(.page-heading__eyebrow) {
        max-width: 760px;
        margin: 16px 0 0;
        color: var(--text-muted);
        line-height: 1.6;
      }

      .documents-layout {
        display: grid;
        grid-template-columns: minmax(320px, 420px) minmax(320px, 420px) minmax(0, 1fr);
        gap: 24px;
        align-items: start;
      }

      .panel {
        min-width: 0;
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        background: var(--surface);
        box-shadow: var(--shadow);
        padding: 24px;
      }

      .metadata-panel {
        position: sticky;
        top: 100px;
      }

      .panel__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .ghost-button {
        display: inline-flex;
        min-height: 42px;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--surface-muted);
        color: var(--text);
        padding: 0 16px;
        font-weight: 800;
        white-space: nowrap;
      }

      .empty-state {
        margin: 20px 0 0;
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius-md);
        color: var(--text-muted);
        padding: 18px;
        line-height: 1.5;
      }

      @media (max-width: 1180px) {
        .documents-layout {
          grid-template-columns: 1fr 1fr;
        }

        .metadata-panel {
          position: static;
          grid-column: 1 / -1;
        }
      }

      @media (max-width: 760px) {
        .page-heading {
          align-items: stretch;
          flex-direction: column;
          padding: 22px;
        }

        .documents-layout {
          grid-template-columns: 1fr;
        }

        .panel {
          padding: 18px;
        }
      }
    `,
  ],
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
