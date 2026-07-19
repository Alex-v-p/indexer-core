import { NgClass, NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import { EvidenceGrade, EvidenceGrading, InformationNeedGrade } from '../../models/query.models';

@Component({
  selector: 'app-evidence-grading-trace',
  standalone: true,
  imports: [NgClass, NgFor, NgIf, StatusBadgeComponent],
  templateUrl: './evidence-grading-trace.component.html',
})
export class EvidenceGradingTraceComponent {
  @Input({ required: true }) grading!: EvidenceGrading;

  percentage(value: number): number {
    return Math.round(Math.min(Math.max(value, 0), 1) * 100);
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

  trackEvidenceGrade(index: number, grade: EvidenceGrade): string {
    return `${grade.evidence_rank}-${index}`;
  }

  trackInformationNeedGrade(index: number, grade: InformationNeedGrade): string {
    return grade.information_need_id || String(index);
  }
}
