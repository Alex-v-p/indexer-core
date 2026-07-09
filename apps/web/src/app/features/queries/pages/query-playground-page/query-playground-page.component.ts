import { NgIf } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { finalize } from 'rxjs';

import { toApiErrorMessage } from '../../../../core/http/api-error';
import { DocumentListComponent } from '../../../../features/documents/components/document-list/document-list.component';
import { DocumentMetadataPanelComponent } from '../../../../features/documents/components/document-metadata-panel/document-metadata-panel.component';
import {
  DocumentUploadComponent,
  DocumentUploadRequest,
} from '../../../../features/documents/components/document-upload/document-upload.component';
import { DocumentsApiService } from '../../../../features/documents/data-access/documents-api.service';
import { DocumentDetail, DocumentSummary } from '../../../../features/documents/models/document.models';
import { AnswerPanelComponent } from '../../components/answer-panel/answer-panel.component';
import { QueryInputComponent } from '../../components/query-input/query-input.component';
import { QueriesApiService } from '../../data-access/queries-api.service';
import { QueryRequest, QueryResponse } from '../../models/query.models';

@Component({
  selector: 'app-query-playground-page',
  standalone: true,
  imports: [
    NgIf,
    DocumentUploadComponent,
    DocumentListComponent,
    DocumentMetadataPanelComponent,
    QueryInputComponent,
    AnswerPanelComponent,
  ],
  template: `
    <section class="workbench" aria-label="Indexer Core minimal RAG UI">
      <aside class="panel documents-panel">
        <div class="panel__header">
          <div>
            <p class="panel__eyebrow">Documents</p>
            <h2>Upload & inspect</h2>
          </div>
          <button class="ghost-button" type="button" (click)="loadDocuments()" [disabled]="documentsLoading()">
            {{ documentsLoading() ? 'Refreshing…' : 'Refresh' }}
          </button>
        </div>

        <app-document-upload
          [uploading]="documentUploading()"
          [error]="documentError()"
          (uploadRequested)="uploadDocument($event)"
        />

        <app-document-list
          [documents]="documents()"
          [loading]="documentsLoading()"
          [selectedDocumentId]="selectedDocument()?.id ?? null"
          (selectDocument)="loadDocumentDetail($event)"
        />

        <app-document-metadata-panel [document]="selectedDocument()" />
      </aside>

      <section class="panel query-panel">
        <div class="panel__header">
          <div>
            <p class="panel__eyebrow">Query graph</p>
            <h2>Ask a grounded question</h2>
          </div>
          <span class="pipeline-pill">baseline</span>
        </div>

        <app-query-input
          [running]="queryRunning()"
          [error]="queryError()"
          (questionAsked)="askQuestion($event)"
        />

        <app-answer-panel *ngIf="queryResult() as result" [result]="result" />
      </section>
    </section>
  `,
  styles: [
    `
      .workbench {
        display: grid;
        grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
        gap: 24px;
        align-items: start;
      }

      .panel {
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        background: var(--surface);
        box-shadow: var(--shadow);
        padding: 24px;
      }

      .documents-panel {
        position: sticky;
        top: 24px;
      }

      .panel__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .panel__eyebrow {
        margin: 0 0 6px;
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      h2 {
        margin: 0;
        font-size: 1.35rem;
        letter-spacing: -0.03em;
      }

      .ghost-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--surface-muted);
        color: var(--text);
        padding: 10px 14px;
        font-weight: 800;
      }

      .pipeline-pill {
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary);
        padding: 6px 12px;
        font-size: 0.8rem;
        font-weight: 800;
      }

      @media (max-width: 1080px) {
        .workbench {
          grid-template-columns: 1fr;
        }

        .documents-panel {
          position: static;
        }
      }

      @media (max-width: 640px) {
        .panel {
          padding: 18px;
        }

        .panel__header {
          align-items: stretch;
          flex-direction: column;
        }
      }
    `,
  ],
})
export class QueryPlaygroundPageComponent implements OnInit {
  private readonly documentsApi = inject(DocumentsApiService);
  private readonly queriesApi = inject(QueriesApiService);

  readonly documents = signal<DocumentSummary[]>([]);
  readonly selectedDocument = signal<DocumentDetail | null>(null);
  readonly documentsLoading = signal(false);
  readonly documentUploading = signal(false);
  readonly queryRunning = signal(false);
  readonly documentError = signal<string | null>(null);
  readonly queryError = signal<string | null>(null);
  readonly queryResult = signal<QueryResponse | null>(null);

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

  askQuestion(request: QueryRequest): void {
    this.queryRunning.set(true);
    this.queryError.set(null);

    this.queriesApi
      .createQueryRun(request)
      .pipe(finalize(() => this.queryRunning.set(false)))
      .subscribe({
        next: (result) => this.queryResult.set(result),
        error: (error: unknown) => this.queryError.set(toApiErrorMessage(error)),
      });
  }
}
