import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import { RetrievalAttempt, RetrievalRetry } from '../../models/query.models';

@Component({
  selector: 'app-retrieval-retry-trace',
  standalone: true,
  imports: [NgFor, NgIf, StatusBadgeComponent],
  templateUrl: './retrieval-retry-trace.component.html',
})
export class RetrievalRetryTraceComponent {
  @Input({ required: true }) retry!: RetrievalRetry;

  label(value: string): string {
    return value.replaceAll('_', ' ');
  }

  percentage(value: number): number {
    return Math.round(Math.min(Math.max(value, 0), 1) * 100);
  }

  trackAttempt(index: number, attempt: RetrievalAttempt): string {
    return `${attempt.attempt_number}-${attempt.pipeline_name}-${index}`;
  }
}
