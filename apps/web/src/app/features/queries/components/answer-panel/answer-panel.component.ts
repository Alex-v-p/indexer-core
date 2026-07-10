import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import { shortId } from '../../../../shared/utils/formatting';
import { QueryResponse, TraceStep } from '../../models/query.models';
import { CitationListComponent } from '../citation-list/citation-list.component';
import { EvidenceViewerComponent } from '../evidence-viewer/evidence-viewer.component';

@Component({
  selector: 'app-answer-panel',
  standalone: true,
  imports: [NgFor, NgIf, StatusBadgeComponent, CitationListComponent, EvidenceViewerComponent],
  templateUrl: './answer-panel.component.html',
})
export class AnswerPanelComponent {
  @Input() result: QueryResponse | null = null;

  readonly shortId = shortId;

  trackTrace(index: number, step: TraceStep): string {
    return step.id ?? `${step.step_order}-${step.name}-${index}`;
  }
}
