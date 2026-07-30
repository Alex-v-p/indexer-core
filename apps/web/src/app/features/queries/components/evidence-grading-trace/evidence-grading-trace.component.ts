import { NgClass, NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import {
  CitationItem,
  EvidenceGrading,
  EvidenceItem,
  InformationNeed,
  InformationNeedAttemptEvidence,
  InformationNeedGrade,
} from '../../models/query.models';
import { TraceEvidenceCardViewModel } from '../../view-models/trace-evidence-view.models';
import {
  buildFinalEvidenceCards,
  placeholderEvidenceCard,
} from '../../utils/trace-evidence-view-model';
import { TraceEvidenceCardComponent } from '../trace-evidence-card/trace-evidence-card.component';

export interface InformationNeedArbitrationView {
  need: InformationNeed;
  grade: InformationNeedGrade | null;
  supportingEvidence: TraceEvidenceCardViewModel[];
  unresolvedExplanation: string | null;
}

@Component({
  selector: 'app-evidence-grading-trace',
  standalone: true,
  imports: [NgClass, NgFor, NgIf, StatusBadgeComponent, TraceEvidenceCardComponent],
  templateUrl: './evidence-grading-trace.component.html',
})
export class EvidenceGradingTraceComponent {
  @Input({ required: true }) grading!: EvidenceGrading;
  @Input() evidence: EvidenceItem[] = [];
  @Input() informationNeeds: InformationNeed[] = [];
  @Input() citations: CitationItem[] = [];
  @Input() attemptEvidence: InformationNeedAttemptEvidence[] = [];

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

  evidenceCards(): TraceEvidenceCardViewModel[] {
    return buildFinalEvidenceCards({
      grading: this.grading,
      evidence: this.evidence ?? [],
      attemptEvidence: this.attemptEvidence ?? [],
      informationNeeds: this.informationNeeds ?? [],
      citations: this.citations ?? [],
    });
  }

  informationNeedViews(): InformationNeedArbitrationView[] {
    const needs = new Map((this.informationNeeds ?? []).map((need) => [need.need_id, need]));
    for (const grade of this.grading.information_need_grades ?? []) {
      if (!needs.has(grade.information_need_id)) {
        needs.set(grade.information_need_id, {
          need_id: grade.information_need_id,
          description: grade.description,
          retrieval_query: 'Retrieval query unavailable for this run',
          required: grade.required,
        });
      }
    }

    const gradesByNeedId = new Map(
      (this.grading.information_need_grades ?? []).map((grade) => [
        grade.information_need_id,
        grade,
      ]),
    );
    const evidenceByRank = new Map(
      this.evidenceCards()
        .filter((item) => item.aggregateRank !== null)
        .map((item) => [item.aggregateRank as number, item]),
    );

    return [...needs.values()].map((need) => {
      const grade = gradesByNeedId.get(need.need_id) ?? null;
      const supportingEvidence = (grade?.supporting_evidence_ranks ?? []).map(
        (rank) =>
          evidenceByRank.get(rank) ??
          placeholderEvidenceCard(
            rank,
            this.grading.grades.find((item) => item.evidence_rank === rank) ?? null,
            this.informationNeeds ?? [],
          ),
      );
      return {
        need,
        grade,
        supportingEvidence,
        unresolvedExplanation: this.unresolvedExplanation(need, grade),
      };
    });
  }

  trackEvidence(index: number, item: TraceEvidenceCardViewModel): string {
    return item.key || String(index);
  }

  trackInformationNeed(index: number, view: InformationNeedArbitrationView): string {
    return view.need.need_id || String(index);
  }

  private unresolvedExplanation(
    need: InformationNeed,
    grade: InformationNeedGrade | null,
  ): string | null {
    if (grade?.status === 'supported') {
      return null;
    }
    if (grade?.rationale) {
      return grade.rationale;
    }
    const needSpecific = (this.grading.unresolved_information ?? []).find(
      (item) => item.includes(need.need_id) || item.includes(need.description),
    );
    if (needSpecific) {
      return needSpecific;
    }
    if (grade === null) {
      return 'No final arbitration grade was stored for this information need.';
    }
    return 'The final arbitration did not record a more specific unresolved explanation.';
  }
}
