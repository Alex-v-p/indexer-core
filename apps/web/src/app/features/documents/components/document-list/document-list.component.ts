import { NgClass, NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge.component';
import { formatBytes } from '../../../../shared/utils/formatting';
import { DocumentSummary } from '../../models/document.models';

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
  @Output() selectDocument = new EventEmitter<string>();

  readonly formatBytes = formatBytes;

  selectedCardClasses(documentId: string): string {
    return this.selectedDocumentId === documentId ? 'border-primary ring-4 ring-primary-soft' : 'border-border';
  }

  trackDocument(_index: number, document: DocumentSummary): string {
    return document.id;
  }
}
