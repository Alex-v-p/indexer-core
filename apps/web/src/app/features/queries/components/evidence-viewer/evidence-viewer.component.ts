import { DecimalPipe, JsonPipe, NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { metadataString } from '../../../../shared/utils/formatting';
import { EvidenceItem } from '../../models/query.models';

@Component({
  selector: 'app-evidence-viewer',
  standalone: true,
  imports: [DecimalPipe, JsonPipe, NgFor, NgIf],
  template: `
    <section class="result-section" *ngIf="evidence.length > 0">
      <div class="section-heading">
        <h4>Evidence</h4>
        <span>{{ evidence.length }}</span>
      </div>
      <details class="evidence-item" *ngFor="let item of evidence; trackBy: trackEvidence">
        <summary>
          <strong>#{{ item.rank }}</strong>
          <span>{{ evidenceSource(item) }}</span>
          <em *ngIf="item.score !== null">score {{ item.score | number: '1.3-4' }}</em>
        </summary>
        <p>{{ item.text }}</p>
        <pre>{{ item.metadata | json }}</pre>
      </details>
    </section>
  `,
  styles: [
    `
      .result-section {
        display: grid;
        gap: 12px;
        margin-top: 6px;
      }

      .section-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      h4 {
        margin: 0;
        font-size: 1rem;
        letter-spacing: -0.03em;
      }

      .section-heading span {
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary);
        padding: 6px 12px;
        font-size: 0.8rem;
        font-weight: 800;
      }

      .evidence-item {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--surface-muted);
        padding: 16px;
      }

      .evidence-item summary {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        cursor: pointer;
        font-weight: 800;
      }

      .evidence-item p {
        margin: 12px 0 0;
        color: var(--text);
        line-height: 1.55;
      }

      pre {
        overflow: auto;
        max-height: 220px;
        margin: 12px 0 0;
        border-radius: var(--radius-sm);
        background: #101828;
        color: #eef4ff;
        padding: 12px;
        font-size: 0.78rem;
      }
    `,
  ],
})
export class EvidenceViewerComponent {
  @Input() evidence: EvidenceItem[] = [];

  trackEvidence(index: number, item: EvidenceItem): string {
    return item.id ?? `${item.rank}-${index}`;
  }

  evidenceSource(item: EvidenceItem): string {
    const filename = metadataString(item.metadata, 'original_filename');
    const section = metadataString(item.metadata, 'section_title');
    return [filename, section].filter(Boolean).join(' · ') || 'Retrieved source chunk';
  }
}
