import { NgClass, NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import { formatBytes } from '../../../../shared/utils/formatting';
import { DocumentSummary } from '../../models/document.models';

export interface DocumentBatchSelectionChange {
  documentId: string;
  selected: boolean;
}

@Component({
  selector: 'app-document-list',
  standalone: true,
  imports: [NgClass, NgFor, NgIf, StatusBadgeComponent],
  templateUrl: './document-list.component.html',
})
export class DocumentListComponent {
  @Input() documents: DocumentSummary[] = [];
  @Input() loading = false;
  @Input() selectedDocumentId: string | null = null;
  @Input() batchSelectedDocumentIds: readonly string[] = [];
  @Input() batchDeleting = false;
  @Output() selectDocument = new EventEmitter<string>();
  @Output() batchSelectionChange = new EventEmitter<DocumentBatchSelectionChange>();

  readonly formatBytes = formatBytes;

  selectedCardClasses(documentId: string): string {
    return this.selectedDocumentId === documentId ? 'border-primary ring-4 ring-primary-soft' : 'border-border';
  }

  isBatchSelected(documentId: string): boolean {
    return this.batchSelectedDocumentIds.includes(documentId);
  }

  trackDocument(_index: number, document: DocumentSummary): string {
    return document.id;
  }
}
