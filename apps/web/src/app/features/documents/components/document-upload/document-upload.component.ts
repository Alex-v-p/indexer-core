import { NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

export interface DocumentUploadRequest {
  file: File;
  title?: string;
}

@Component({
  selector: 'app-document-upload',
  standalone: true,
  imports: [FormsModule, NgIf],
  templateUrl: './document-upload.component.html',
})
export class DocumentUploadComponent {
  @Input() uploading = false;
  @Input() error: string | null = null;
  @Output() uploadRequested = new EventEmitter<DocumentUploadRequest>();

  readonly selectedFile = signal<File | null>(null);
  title = '';

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFile.set(input.files?.item(0) ?? null);
  }

  submitUpload(): void {
    const file = this.selectedFile();
    if (!file) {
      return;
    }

    this.uploadRequested.emit({ file, title: this.title });
    this.title = '';
    this.selectedFile.set(null);
  }
}
