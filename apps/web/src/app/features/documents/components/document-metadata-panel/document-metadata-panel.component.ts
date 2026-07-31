import { NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import { formatDate } from '../../../../shared/utils/formatting';
import { ChunkIndex, DocumentDetail, DocumentVersion } from '../../models/document.models';

@Component({
  selector: 'app-document-metadata-panel',
  standalone: true,
  imports: [NgFor, NgIf, StatusBadgeComponent],
  templateUrl: './document-metadata-panel.component.html',
})
export class DocumentMetadataPanelComponent {
  @Input() document: DocumentDetail | null = null;
  @Input() deleting = false;
  @Output() deleteRequested = new EventEmitter<DocumentDetail>();

  readonly formatDate = formatDate;

  trackChunk(_index: number, chunk: ChunkIndex): string {
    return chunk.id;
  }

  trackVersion(_index: number, version: DocumentVersion): string {
    return version.id;
  }
}
