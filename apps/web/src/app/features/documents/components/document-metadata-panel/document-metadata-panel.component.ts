import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge.component';
import { formatDate } from '../../../../shared/utils/formatting';
import { ChunkIndex, DocumentDetail } from '../../models/document.models';

@Component({
  selector: 'app-document-metadata-panel',
  standalone: true,
  imports: [NgFor, NgIf, StatusBadgeComponent],
  template: `
    <article class="detail-card" *ngIf="document as doc">
      <div class="detail-card__header">
        <div>
          <p class="panel__eyebrow">Selected document</p>
          <h3>{{ doc.title }}</h3>
        </div>
        <app-status-badge [status]="doc.status" />
      </div>

      <dl class="metadata-grid">
        <div>
          <dt>Created</dt>
          <dd>{{ formatDate(doc.created_at) }}</dd>
        </div>
        <div>
          <dt>Chunks</dt>
          <dd>{{ doc.chunks.length }}</dd>
        </div>
        <div>
          <dt>Versions</dt>
          <dd>{{ doc.versions.length }}</dd>
        </div>
        <div>
          <dt>Checksum</dt>
          <dd class="truncate">{{ doc.checksum_sha256 || '—' }}</dd>
        </div>
      </dl>

      <details class="chunk-preview" *ngIf="doc.chunks.length > 0">
        <summary>Preview chunk index metadata</summary>
        <ol>
          <li *ngFor="let chunk of doc.chunks.slice(0, 5); trackBy: trackChunk">
            <strong>#{{ chunk.ordinal }}</strong>
            <span>{{ chunk.section_title || 'No section title' }}</span>
            <small>
              {{ chunk.source_page_start ? 'Page ' + chunk.source_page_start : 'No page' }} ·
              {{ chunk.token_count || 0 }} tokens
            </small>
          </li>
        </ol>
      </details>
    </article>
  `,
  styles: [
    `
      .detail-card {
        display: grid;
        gap: 16px;
        margin-top: 20px;
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--surface-muted);
        padding: 16px;
      }

      .detail-card__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .panel__eyebrow {
        margin: 0 0 6px;
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      h3 {
        margin: 0;
        font-size: 1.05rem;
        letter-spacing: -0.03em;
      }

      .metadata-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 0;
      }

      .metadata-grid div {
        min-width: 0;
        border-radius: var(--radius-sm);
        background: var(--surface);
        padding: 12px;
      }

      dt {
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      dd {
        margin: 6px 0 0;
        font-weight: 700;
      }

      .truncate {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .chunk-preview summary {
        cursor: pointer;
        font-weight: 800;
      }

      .chunk-preview ol {
        display: grid;
        gap: 10px;
        margin: 12px 0 0;
        padding: 0;
        list-style: none;
      }

      .chunk-preview li {
        display: grid;
        gap: 4px;
      }

      .chunk-preview small {
        color: var(--text-muted);
        font-size: 0.83rem;
      }

      @media (max-width: 640px) {
        .detail-card__header {
          align-items: stretch;
          flex-direction: column;
        }

        .metadata-grid {
          grid-template-columns: 1fr;
        }
      }
    `,
  ],
})
export class DocumentMetadataPanelComponent {
  @Input() document: DocumentDetail | null = null;

  readonly formatDate = formatDate;

  trackChunk(_index: number, chunk: ChunkIndex): string {
    return chunk.id;
  }
}
