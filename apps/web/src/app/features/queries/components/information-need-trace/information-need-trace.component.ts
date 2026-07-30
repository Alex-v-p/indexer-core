import { NgClass, NgFor, NgIf, PercentPipe } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import {
  InformationNeedAttempt,
  InformationNeedExecution,
  InformationNeedResolution,
} from '../../models/query.models';

@Component({
  selector: 'app-information-need-trace',
  standalone: true,
  imports: [NgClass, NgFor, NgIf, PercentPipe, StatusBadgeComponent],
  templateUrl: './information-need-trace.component.html',
})
export class InformationNeedTraceComponent {
  @Input() resolution: InformationNeedResolution | null = null;

  trackExecution(index: number, execution: InformationNeedExecution): string {
    return execution.information_need_id || String(index);
  }

  trackAttempt(index: number, attempt: InformationNeedAttempt): string {
    return `${attempt.attempt_number}-${attempt.pipeline_name}-${index}`;
  }

  percentage(value: number): number {
    return Math.round(Math.min(Math.max(value, 0), 1) * 100);
  }

  label(value: string): string {
    return value.replaceAll('_', ' ');
  }

  coverageClass(value: number): string {
    if (value >= 0.75) {
      return 'bg-success';
    }
    if (value >= 0.4) {
      return 'bg-warning';
    }
    return 'bg-danger';
  }
}
