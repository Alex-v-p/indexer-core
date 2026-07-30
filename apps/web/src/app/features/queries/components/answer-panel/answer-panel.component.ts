import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import { shortId } from '../../../../shared/utils/formatting';
import { QueryResponse } from '../../models/query.models';
import {
  QueryAnswerViewModel,
  buildQueryAnswerViewModel,
} from '../../view-models/answer.view-model';
import { AgentTraceComponent } from '../agent-trace/agent-trace.component';
import { CitationListComponent } from '../citation-list/citation-list.component';
import { EvidenceViewerComponent } from '../evidence-viewer/evidence-viewer.component';

@Component({
  selector: 'app-answer-panel',
  standalone: true,
  imports: [
    NgIf,
    NgFor,
    StatusBadgeComponent,
    AgentTraceComponent,
    CitationListComponent,
    EvidenceViewerComponent,
  ],
  templateUrl: './answer-panel.component.html',
})
export class AnswerPanelComponent {
  @Input() result: QueryResponse | null = null;

  readonly shortId = shortId;

  get answerViewModel(): QueryAnswerViewModel | null {
    return this.result ? buildQueryAnswerViewModel(this.result) : null;
  }
}
