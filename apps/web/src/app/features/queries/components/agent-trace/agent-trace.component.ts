import { NgFor, NgIf, PercentPipe } from '@angular/common';
import { Component, Input } from '@angular/core';

import { AgentTraceViewModel } from '../../models/agent-trace-view.models';
import {
  DocumentPreference,
  InformationNeed,
  QueryResponse,
} from '../../models/query.models';
import { buildAgentTraceViewModel } from '../../utils/agent-trace-view-model';
import { AgentJourneyOverviewComponent } from '../agent-journey-overview/agent-journey-overview.component';
import { DocumentPreferenceTraceComponent } from '../document-preference-trace/document-preference-trace.component';
import { EvidenceGradingTraceComponent } from '../evidence-grading-trace/evidence-grading-trace.component';
import { ExecutionTraceComponent } from '../execution-trace/execution-trace.component';
import { InformationNeedTraceComponent } from '../information-need-trace/information-need-trace.component';
import { RetrievalRetryTraceComponent } from '../retrieval-retry-trace/retrieval-retry-trace.component';

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
    RetrievalRetryTraceComponent,
  ],
  templateUrl: './agent-trace.component.html',
})
export class AgentTraceComponent {
  private currentResult!: QueryResponse;

  traceView: AgentTraceViewModel | null = null;

  @Input({ required: true })
  set result(value: QueryResponse) {
    this.currentResult = value;
    this.traceView = buildAgentTraceViewModel(value);
  }

  get result(): QueryResponse {
    return this.currentResult;
  }

  label(value: string): string {
    return value.replaceAll('_', ' ');
  }

  percentage(value: number): number {
    return Math.round(Math.min(Math.max(value, 0), 1) * 100);
  }

  strategiesUsedLabel(): string {
    const strategies = this.traceView?.summary.strategiesUsed ?? [];
    return strategies.length > 0
      ? strategies.map((strategy) => this.label(strategy)).join(', ')
      : 'No strategies recorded';
  }

  pipelinesUsedLabel(): string {
    const pipelines = this.traceView?.summary.pipelinesUsed ?? [];
    return pipelines.length > 0 ? pipelines.join(', ') : 'No pipelines recorded';
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

  hasDocumentPreferenceTrace(): boolean {
    return Boolean(
      this.primaryDocumentPreference() ||
        this.result.trace.some(
          (step) =>
            step.name === 'detect_primary_document' ||
            step.metadata['primary_document_detection'] ||
            step.metadata['active_information_need_lookup'],
        ),
    );
  }

  hasFinalEvidenceArbitration(): boolean {
    return this.result.trace.some(
      (step) => step.name === 'arbitrate_final_evidence' || Boolean(step.metadata['evidence_arbitration']),
    );
  }

  totalAttempts(): number {
    return this.traceView?.summary.totalAttempts ?? (this.result.evidence.length > 0 ? 1 : 0);
  }

  retryCount(): number {
    return this.traceView?.summary.retries ?? 0;
  }

  overallCoverage(): number | null {
    if (this.result.evidence_grading) {
      return this.result.evidence_grading.coverage_score;
    }

    const grades = (this.result.information_need_resolution?.executions ?? [])
      .map((execution) => execution.final_grade?.coverage_score)
      .filter((value): value is number => value !== undefined);

    if (grades.length === 0) {
      return null;
    }
    return grades.reduce((sum, value) => sum + value, 0) / grades.length;
  }

  gradingStatus(): string {
    if (this.result.evidence_grading) {
      return this.result.evidence_grading.status;
    }
    if (this.result.information_need_resolution?.complete) {
      return 'sufficient';
    }
    if (this.result.information_need_resolution) {
      return 'partial';
    }
    return 'not graded';
  }

  resolvedNeedsLabel(): string {
    const resolution = this.result.information_need_resolution;
    if (!resolution) {
      const count = this.result.information_need_decomposition?.information_need_count ?? 0;
      return count > 0 ? `0/${count}` : '—';
    }
    return `${resolution.supported_information_need_count}/${resolution.information_need_count}`;
  }

  hasMetadataScope(): boolean {
    const classification = this.result.classification;
    return Boolean(
      classification?.document_constraint.active ||
        classification?.version_constraint.active ||
        classification?.date_constraints.length,
    );
  }

  trackNeed(index: number, need: InformationNeed): string {
    return need.need_id || String(index);
  }
}
