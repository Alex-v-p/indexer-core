import { NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

import {
  DocumentSubjectDecision,
  Subject,
  SubjectDecisionState,
} from '../../../subjects/models/subject.models';
import { SubjectClassificationStatus } from '../../models/document.models';

export interface DocumentSubjectChangeRequest {
  subjectId: string;
  state: Extract<SubjectDecisionState, 'assigned' | 'rejected'>;
  expectedRevision: number;
}

export interface DocumentSuggestionReviewRequest {
  subjectId: string;
  decision: 'accept' | 'reject';
  expectedRevision: number;
}

@Component({
  selector: 'app-document-subject-panel',
  standalone: true,
  imports: [NgFor, NgIf],
  templateUrl: './document-subject-panel.component.html',
})
export class DocumentSubjectPanelComponent {
  @Input() subjects: Subject[] = [];
  @Input() decisions: DocumentSubjectDecision[] = [];
  @Input() busy = false;
  @Input() loading = false;
  @Input() classificationStatus: SubjectClassificationStatus | null = null;
  @Input() classificationBusy = false;
  @Output() decisionRequested = new EventEmitter<DocumentSubjectChangeRequest>();
  @Output() suggestionReviewRequested = new EventEmitter<DocumentSuggestionReviewRequest>();
  @Output() reclassifyRequested = new EventEmitter<void>();

  get classificationIsRunning(): boolean {
    return this.classificationBusy || this.classificationStatus?.status === 'queued' || this.classificationStatus?.status === 'running';
  }

  requestReclassification(): void {
    if (!this.classificationIsRunning) {
      this.reclassifyRequested.emit();
    }
  }

  decisionFor(subjectId: string): DocumentSubjectDecision | null {
    return this.decisions.find((decision) => decision.subject_id === subjectId) ?? null;
  }

  requestState(subjectId: string, state: 'assigned' | 'rejected'): void {
    const current = this.decisionFor(subjectId);
    this.decisionRequested.emit({
      subjectId,
      state,
      expectedRevision: current?.revision ?? 0,
    });
  }

  reviewSuggestion(
    decision: DocumentSubjectDecision,
    review: 'accept' | 'reject',
  ): void {
    this.suggestionReviewRequested.emit({
      subjectId: decision.subject_id,
      decision: review,
      expectedRevision: decision.revision,
    });
  }
}
