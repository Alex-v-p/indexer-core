import { DecimalPipe, NgClass, NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { CitationItem } from '../../models/query.models';
import {
  TraceEvidenceCardViewModel,
  TraceEvidenceNeedReference,
} from '../../view-models/trace-evidence-view.models';

@Component({
  selector: 'app-trace-evidence-card',
  standalone: true,
  imports: [DecimalPipe, NgClass, NgFor, NgIf],
  templateUrl: './trace-evidence-card.component.html',
})
export class TraceEvidenceCardComponent {
  @Input({ required: true }) item!: TraceEvidenceCardViewModel;

  percentage(value: number | null): number | null {
    return value === null ? null : Math.round(Math.min(Math.max(value, 0), 1) * 100);
  }

  trackNeed(index: number, need: TraceEvidenceNeedReference): string {
    return need.needId || String(index);
  }

  trackCitation(index: number, citation: CitationItem): string {
    return citation.id ?? String(index);
  }
}
