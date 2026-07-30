import { JsonPipe, NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import { TraceStep } from '../../models/query.models';

@Component({
  selector: 'app-execution-trace',
  standalone: true,
  imports: [JsonPipe, NgFor, NgIf, StatusBadgeComponent],
  templateUrl: './execution-trace.component.html',
})
export class ExecutionTraceComponent {
  @Input() trace: TraceStep[] = [];

  trackTrace(index: number, step: TraceStep): string {
    return step.id ?? `${step.step_order}-${step.name}-${index}`;
  }

  graphName(step: TraceStep): string {
    const graphName = step.metadata['graph_name'];
    return typeof graphName === 'string' && graphName.trim().length > 0
      ? graphName.replaceAll('_', ' ')
      : 'query graph';
  }

  graphDepth(step: TraceStep): number {
    const depth = step.metadata['graph_depth'];
    return typeof depth === 'number' && Number.isFinite(depth) ? depth : 0;
  }

  nodeName(step: TraceStep): string {
    return step.name.replaceAll('_', ' ');
  }

  stepType(step: TraceStep): string {
    return (step.step_type ?? 'node').replaceAll('_', ' ');
  }
}
