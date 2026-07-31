import { NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import { formatDate } from '../../../../shared/utils/formatting';
import { ChunkIndex, DocumentDetail, DocumentVersion } from '../../models/document.models';

export interface DeleteDocumentVersionsRequest {
  document: DocumentDetail;
  versions: DocumentVersion[];
}

@Component({
  selector: 'app-document-metadata-panel',
  standalone: true,
  imports: [NgFor, NgIf, StatusBadgeComponent],
  templateUrl: './document-metadata-panel.component.html',
})
export class DocumentMetadataPanelComponent implements OnChanges {
  @Input() document: DocumentDetail | null = null;
  @Input() deleting = false;
  @Output() deleteVersionsRequested = new EventEmitter<DeleteDocumentVersionsRequest>();

  readonly formatDate = formatDate;
  private readonly selectedVersionIds = new Set<string>();

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['document']) {
      this.selectedVersionIds.clear();
    }
  }

  get selectedVersionCount(): number {
    return this.selectedVersionIds.size;
  }

  isVersionSelected(versionId: string): boolean {
    return this.selectedVersionIds.has(versionId);
  }

  toggleVersion(versionId: string, checked: boolean): void {
    if (checked) {
      this.selectedVersionIds.add(versionId);
    } else {
      this.selectedVersionIds.delete(versionId);
    }
  }

  requestSingleDeletion(version: DocumentVersion): void {
    if (this.document) {
      this.deleteVersionsRequested.emit({ document: this.document, versions: [version] });
    }
  }

  requestSelectedDeletion(): void {
    if (!this.document) {
      return;
    }
    const versions = this.document.versions.filter((version) =>
      this.selectedVersionIds.has(version.id),
    );
    if (versions.length > 0) {
      this.deleteVersionsRequested.emit({ document: this.document, versions });
    }
  }

  trackChunk(_index: number, chunk: ChunkIndex): string {
    return chunk.id;
  }

  trackVersion(_index: number, version: DocumentVersion): string {
    return version.id;
  }
}
