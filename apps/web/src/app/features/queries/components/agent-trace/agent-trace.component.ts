import { NgFor, NgIf, PercentPipe } from '@angular/common';
import { Component, Input } from '@angular/core';

import {
  CitationItem,
  DocumentPreference,
  EvidenceItem,
  InformationNeed,
  InformationNeedAttemptEvidence,
  QueryResponse,
  TraceStep,
} from '../../models/query.models';
import { AgentJourneyOverviewComponent } from '../agent-journey-overview/agent-journey-overview.component';
import { DocumentPreferenceTraceComponent } from '../document-preference-trace/document-preference-trace.component';
import { EvidenceGradingTraceComponent } from '../evidence-grading-trace/evidence-grading-trace.component';
import { ExecutionTraceComponent } from '../execution-trace/execution-trace.component';
import { InformationNeedTraceComponent } from '../information-need-trace/information-need-trace.component';
import { RetrievalStrategyEvolutionComponent } from '../retrieval-strategy-evolution/retrieval-strategy-evolution.component';

@Component({
  selector: 'app-agent-trace',
  standalone: true,
  imports: [
    PercentPipe,
    NgFor,
    NgIf,
    AgentJourneyOverviewComponent,
    DocumentPreferenceTraceComponent,
    EvidenceGradingTraceComponent,
    ExecutionTraceComponent,
    InformationNeedTraceComponent,
    RetrievalStrategyEvolutionComponent,
  ],
  templateUrl: './agent-trace.component.html',
})
export class AgentTraceComponent {
  @Input({ required: true }) result!: QueryResponse;

  label(value: string): string {
    return value.replaceAll('_', ' ');
  }

  primaryDocumentPreference(): DocumentPreference | null {
    if (this.result.primary_document_preference) {
      return this.result.primary_document_preference;
    }
    const executions = this.result.information_need_resolution?.executions ?? [];
    for (let executionIndex = executions.length - 1; executionIndex >= 0; executionIndex -= 1) {
      const execution = executions[executionIndex];
      const attempts = execution.attempts ?? [];
      for (let attemptIndex = attempts.length - 1; attemptIndex >= 0; attemptIndex -= 1) {
        const preference = attempts[attemptIndex].plan?.preferred_document;
        if (preference) {
          return preference;
        }
      }
      const plans = execution.plan_history ?? [];
      for (let planIndex = plans.length - 1; planIndex >= 0; planIndex -= 1) {
        const preference = plans[planIndex].preferred_document;
        if (preference) {
          return preference;
        }
      }
    }
    return null;
  }

  informationNeeds(): InformationNeed[] {
    const needs = new Map<string, InformationNeed>();
    for (const need of this.result.information_need_decomposition?.information_needs ?? []) {
      needs.set(need.need_id, need);
    }
    for (const execution of this.result.information_need_resolution?.executions ?? []) {
      needs.set(execution.information_need_id, execution.information_need);
    }
    return [...needs.values()];
  }

  hasRetrievalStrategyEvolution(): boolean {
    if (this.result.retrieval_plan) {
      return true;
    }
    return (this.result.information_need_resolution?.executions ?? []).some(
      (execution) =>
        (execution.attempts ?? []).length > 0 || (execution.plan_history ?? []).length > 0,
    );
  }

  hasDocumentPreferenceTrace(): boolean {
    return Boolean(
      this.primaryDocumentPreference() ||
        this.result.information_need_resolution?.executions.some(
          (execution) => (execution.attempts ?? []).length > 0,
        ) ||
        this.traceSteps().some(
          (step) =>
            step.name === 'detect_primary_document' ||
            step.metadata['primary_document_detection'] ||
            step.metadata['active_information_need_lookup'],
        ),
    );
  }

  attemptEvidence(): InformationNeedAttemptEvidence[] {
    return (this.result.information_need_resolution?.executions ?? []).flatMap((execution) =>
      (execution.attempts ?? []).flatMap((attempt) => attempt.evidence ?? []),
    );
  }

  hasFinalEvidenceArbitration(): boolean {
    return this.traceSteps().some(
      (step) =>
        step.name === 'arbitrate_final_evidence' ||
        Boolean(step.metadata['evidence_arbitration']),
    );
  }

  hasMetadataScope(): boolean {
    const classification = this.result.classification;
    return Boolean(
      classification?.document_constraint.active ||
        classification?.version_constraint.active ||
        classification?.date_constraints.length,
    );
  }

  evidenceItems(): EvidenceItem[] {
    return this.result.evidence ?? [];
  }

  citationItems(): CitationItem[] {
    return this.result.citations ?? [];
  }

  traceSteps(): TraceStep[] {
    return this.result.trace ?? [];
  }

  trackNeed(index: number, need: InformationNeed): string {
    return need.need_id || String(index);
  }
}
