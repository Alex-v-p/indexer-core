import { NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge.component';
import { formatBytes } from '../../../../shared/utils/formatting';
import { DocumentSummary } from '../../models/document.models';

@Component({
  selector: 'app-document-list',
  standalone: true,
  imports: [NgFor, NgIf, StatusBadgeComponent],
  template: `
    <div class="empty-state" *ngIf="!loading && documents.length === 0">
      No documents indexed yet. Upload one to make the baseline retriever useful.
    </div>

    <div class="document-list" *ngIf="documents.length > 0">
      <button
        class="document-card"
        type="button"
        *ngFor="let document of documents; trackBy: trackDocument"
        [class.document-card--selected]="selectedDocumentId === document.id"
        (click)="selectDocument.emit(document.id)"
      >
        <span class="document-card__topline">
          <strong>{{ document.title }}</strong>
          <app-status-badge [status]="document.status" />
        </span>
        <span class="document-card__meta">
          {{ document.original_filename || document.content_type || 'Untitled source' }}
        </span>
        <span class="document-card__footer">
          <span>{{ document.chunk_count }} chunks</span>
          <span>{{ formatBytes(document.size_bytes) }}</span>
        </span>
      </button>
    </div>
  `,
  styles: [
    `
      .empty-state {
        margin-top: 20px;
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius-md);
        color: var(--text-muted);
        padding: 18px;
        line-height: 1.5;
      }

      .document-list {
        display: grid;
        gap: 12px;
        margin-top: 20px;
      }

      .document-card {
        display: grid;
        gap: 10px;
        width: 100%;
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--surface);
        color: inherit;
        padding: 16px;
        text-align: left;
      }

      .document-card--selected {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px var(--primary-soft);
      }

      .document-card__topline,
      .document-card__footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .document-card__topline strong {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .document-card__meta,
      .document-card__footer {
        color: var(--text-muted);
        font-size: 0.83rem;
      }
    `,
  ],
})
export class DocumentListComponent {
  @Input() documents: DocumentSummary[] = [];
  @Input() loading = false;
  @Input() selectedDocumentId: string | null = null;
  @Output() selectDocument = new EventEmitter<string>();

  readonly formatBytes = formatBytes;

  trackDocument(_index: number, document: DocumentSummary): string {
    return document.id;
  }
}
