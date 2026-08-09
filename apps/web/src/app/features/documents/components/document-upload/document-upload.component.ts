import { NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { DocumentSummary, DocumentUploadMode, DocumentUploadRequest } from '../../models/document.models';

@Component({
  selector: 'app-document-upload',
  standalone: true,
  imports: [FormsModule, NgFor, NgIf],
  templateUrl: './document-upload.component.html',
})
export class DocumentUploadComponent {
  @Input() uploading = false;
  @Input() error: string | null = null;
  @Input() documents: DocumentSummary[] = [];
  @Output() uploadRequested = new EventEmitter<DocumentUploadRequest>();

  readonly selectedFiles = signal<File[]>([]);
  title = '';
  publishedAt = '';
  uploadMode: DocumentUploadMode = 'automatic';
  versionOfDocumentId = '';

  onFilesSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFiles.set(Array.from(input.files ?? []));
  }

  submitUpload(): void {
    const files = this.selectedFiles();
    if (files.length === 0 || (this.uploadMode === 'manual_version' && !this.versionOfDocumentId)) {
      return;
    }

    this.uploadRequested.emit({
      files,
      title: files.length === 1 ? this.title.trim() || undefined : undefined,
      publishedAt: this.publishedAt || undefined,
      detectExistingVersions: this.uploadMode === 'automatic',
      versionOfDocumentId:
        this.uploadMode === 'manual_version' ? this.versionOfDocumentId : undefined,
      subjectIds: [],
    });
    this.title = '';
    this.publishedAt = '';
    this.selectedFiles.set([]);
  }
}
