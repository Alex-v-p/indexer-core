import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge.component';
import { formatDate } from '../../../../shared/utils/formatting';
import { ChunkIndex, DocumentDetail } from '../../models/document.models';

@Component({
  selector: 'app-document-metadata-panel',
  standalone: true,
  imports: [NgFor, NgIf, StatusBadgeComponent],
  templateUrl: './document-metadata-panel.component.html',
})
export class DocumentMetadataPanelComponent {
  @Input() document: DocumentDetail | null = null;

  readonly formatDate = formatDate;

  trackChunk(_index: number, chunk: ChunkIndex): string {
    return chunk.id;
  }
}
