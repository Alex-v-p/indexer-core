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
  template: `
    <form class="upload-card" (ngSubmit)="submitUpload()">
      <label>
        <span>Optional title</span>
        <input
          type="text"
          name="documentTitle"
          [(ngModel)]="title"
          placeholder="e.g. Architecture notes"
        />
      </label>

      <label class="file-picker">
        <span>Source file</span>
        <input
          type="file"
          accept=".pdf,.txt,.md,text/plain,text/markdown,application/pdf"
          (change)="onFileSelected($event)"
        />
        <strong>{{ selectedFile()?.name || 'Choose a PDF, text, or markdown file' }}</strong>
      </label>

      <button class="primary-button" type="submit" [disabled]="!selectedFile() || uploading">
        {{ uploading ? 'Uploading…' : 'Upload document' }}
      </button>

      <p class="error-message" *ngIf="error">{{ error }}</p>
    </form>
  `,
  styles: [
    `
      .upload-card {
        display: grid;
        gap: 16px;
        margin-top: 20px;
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--surface-muted);
        padding: 16px;
      }

      label {
        display: grid;
        gap: 8px;
        color: var(--text-muted);
        font-size: 0.86rem;
        font-weight: 700;
      }

      input {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        background: var(--surface);
        color: var(--text);
        padding: 12px 14px;
        outline: none;
        transition: border-color 160ms ease, box-shadow 160ms ease;
      }

      input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px var(--primary-soft);
      }

      .file-picker input {
        padding: 10px;
      }

      .file-picker strong {
        overflow: hidden;
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 700;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .primary-button {
        display: inline-flex;
        min-height: 44px;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: var(--primary);
        color: white;
        padding: 0 20px;
        font-weight: 800;
        transition: background 160ms ease, transform 160ms ease;
      }

      .primary-button:not(:disabled):hover {
        background: var(--primary-strong);
        transform: translateY(-1px);
      }

      .error-message {
        margin: 0;
        color: var(--danger);
        font-weight: 700;
      }
    `,
  ],
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
