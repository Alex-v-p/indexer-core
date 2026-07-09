import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { metadataString } from '../../../../shared/utils/formatting';
import { CitationItem } from '../../models/query.models';

@Component({
  selector: 'app-citation-list',
  standalone: true,
  imports: [NgFor, NgIf],
  template: `
    <section class="result-section" *ngIf="citations.length > 0">
      <div class="section-heading">
        <h4>Citations</h4>
        <span>{{ citations.length }}</span>
      </div>
      <div class="citation-grid">
        <article class="citation-card" *ngFor="let citation of citations; trackBy: trackCitation">
          <div class="citation-card__label">
            <strong>{{ citation.label || '[' + citation.citation_index + ']' }}</strong>
            <span *ngIf="citation.page_number">Page {{ citation.page_number }}</span>
          </div>
          <p>{{ citation.quote || 'No quote captured.' }}</p>
          <small>{{ citationSource(citation) }}</small>
        </article>
      </div>
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

      .citation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 12px;
      }

      .citation-card {
        display: grid;
        gap: 8px;
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--surface-muted);
        padding: 16px;
      }

      .citation-card__label {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: var(--primary);
      }

      .citation-card p {
        margin: 0;
        color: var(--text);
        line-height: 1.55;
      }

      .citation-card small {
        color: var(--text-muted);
        font-size: 0.83rem;
      }
    `,
  ],
})
export class CitationListComponent {
  @Input() citations: CitationItem[] = [];

  trackCitation(index: number, citation: CitationItem): string {
    return citation.id ?? `${citation.citation_index}-${index}`;
  }

  citationSource(citation: CitationItem): string {
    const filename = metadataString(citation.metadata, 'original_filename');
    const section = metadataString(citation.metadata, 'section_title');
    return [filename, section].filter(Boolean).join(' · ') || 'Retrieved source chunk';
  }
}
