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

  readonly selectedFile = signal<File | null>(null);
  title = '';
  publishedAt = '';
  uploadMode: DocumentUploadMode = 'automatic';
  versionOfDocumentId = '';

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFile.set(input.files?.item(0) ?? null);
  }

  submitUpload(): void {
    const file = this.selectedFile();
    if (!file || (this.uploadMode === 'manual_version' && !this.versionOfDocumentId)) {
      return;
    }

    this.uploadRequested.emit({
      file,
      title: this.title.trim() || undefined,
      publishedAt: this.publishedAt || undefined,
      detectExistingVersions: this.uploadMode === 'automatic',
      versionOfDocumentId:
        this.uploadMode === 'manual_version' ? this.versionOfDocumentId : undefined,
    });
    this.title = '';
    this.publishedAt = '';
    this.selectedFile.set(null);
  }
}
