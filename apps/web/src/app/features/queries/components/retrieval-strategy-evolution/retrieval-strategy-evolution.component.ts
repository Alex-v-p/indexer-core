import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import {
  AgentJourneyAttempt,
  AgentTraceViewModel,
  StrategyEvolutionLane,
} from '../../models/agent-trace-view.models';
import { QueryResponse } from '../../models/query.models';
import {
  buildAgentTraceViewModel,
  displayLabel,
} from '../../utils/agent-trace-view-model';

@Component({
  selector: 'app-retrieval-strategy-evolution',
  standalone: true,
  imports: [NgFor, NgIf],
  templateUrl: './retrieval-strategy-evolution.component.html',
})
export class RetrievalStrategyEvolutionComponent {
  viewModel: AgentTraceViewModel | null = null;

  @Input({ required: true })
  set result(value: QueryResponse) {
    this.viewModel = buildAgentTraceViewModel(value);
  }

  evolutions(): StrategyEvolutionLane[] {
    return (this.viewModel?.strategyEvolution ?? []).filter(
      (evolution) => evolution.attempts.length > 0,
    );
  }

  label(value: string | null | undefined): string {
    return displayLabel(value);
  }

  trackEvolution(index: number, evolution: StrategyEvolutionLane): string {
    return evolution.informationNeedId || String(index);
  }

  trackAttempt(index: number, attempt: AgentJourneyAttempt): string {
    return `${attempt.attemptNumber}-${attempt.pipeline}-${index}`;
  }
}
