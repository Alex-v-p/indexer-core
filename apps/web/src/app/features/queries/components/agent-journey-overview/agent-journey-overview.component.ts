import { NgClass, NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import {
  AgentJourneyAttempt,
  AgentJourneyNeedLane,
  AgentTraceViewModel,
} from '../../models/agent-trace-view.models';
import { QueryResponse } from '../../models/query.models';
import {
  buildAgentTraceViewModel,
  displayLabel,
} from '../../utils/agent-trace-view-model';

@Component({
  selector: 'app-agent-journey-overview',
  standalone: true,
  imports: [NgClass, NgFor, NgIf, StatusBadgeComponent],
  templateUrl: './agent-journey-overview.component.html',
})
export class AgentJourneyOverviewComponent {
  private currentResult!: QueryResponse;

  viewModel: AgentTraceViewModel | null = null;

  @Input({ required: true })
  set result(value: QueryResponse) {
    this.currentResult = value;
    this.viewModel = buildAgentTraceViewModel(value);
  }

  get result(): QueryResponse {
    return this.currentResult;
  }

  label(value: string | null | undefined): string {
    return displayLabel(value);
  }

  percentage(value: number | null): number | null {
    if (value === null) {
      return null;
    }
    return Math.round(Math.min(Math.max(value, 0), 1) * 100);
  }

  strategiesLabel(viewModel: AgentTraceViewModel): string {
    return viewModel.summary.strategiesUsed.length > 0
      ? viewModel.summary.strategiesUsed.map((value) => this.label(value)).join(', ')
      : 'None recorded';
  }

  pipelinesLabel(viewModel: AgentTraceViewModel): string {
    return viewModel.summary.pipelinesUsed.length > 0
      ? viewModel.summary.pipelinesUsed.join(', ')
      : 'None recorded';
  }

  isLastAttempt(lane: AgentJourneyNeedLane, attempt: AgentJourneyAttempt): boolean {
    return lane.attempts.at(-1) === attempt;
  }

  trackLane(index: number, lane: AgentJourneyNeedLane): string {
    return lane.need.need_id || String(index);
  }

  trackAttempt(index: number, attempt: AgentJourneyAttempt): string {
    return `${attempt.attemptNumber}-${attempt.pipeline}-${index}`;
  }

  strategyClass(strategy: string): string {
    switch (strategy) {
      case 'multi_query':
        return 'border-primary/25 bg-primary-soft text-primary';
      case 'rerank':
        return 'border-warning/25 bg-warning/10 text-warning';
      case 'hierarchical':
        return 'border-success/25 bg-success/10 text-success';
      default:
        return 'border-border bg-surface text-text-muted';
    }
  }
}
