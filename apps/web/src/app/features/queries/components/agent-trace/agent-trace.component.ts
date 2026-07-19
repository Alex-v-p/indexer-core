import { NgFor, NgIf, PercentPipe } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge/status-badge.component';
import {
  DocumentPreference,
  InformationNeed,
  InformationNeedRetrievalPlan,
  QueryResponse,
  RetrievalPlan,
} from '../../models/query.models';
import { DocumentPreferenceTraceComponent } from '../document-preference-trace/document-preference-trace.component';
import { EvidenceGradingTraceComponent } from '../evidence-grading-trace/evidence-grading-trace.component';
import { ExecutionTraceComponent } from '../execution-trace/execution-trace.component';
import { InformationNeedTraceComponent } from '../information-need-trace/information-need-trace.component';
import { RetrievalRetryTraceComponent } from '../retrieval-retry-trace/retrieval-retry-trace.component';

type AgentRetrievalPlan = RetrievalPlan | InformationNeedRetrievalPlan;

@Component({
  selector: 'app-agent-trace',
  standalone: true,
  imports: [
    PercentPipe,
    NgFor,
    NgIf,
    StatusBadgeComponent,
    DocumentPreferenceTraceComponent,
    EvidenceGradingTraceComponent,
    ExecutionTraceComponent,
    InformationNeedTraceComponent,
    RetrievalRetryTraceComponent,
  ],
  templateUrl: './agent-trace.component.html',
})
export class AgentTraceComponent {
  @Input({ required: true }) result!: QueryResponse;

  label(value: string): string {
    return value.replaceAll('_', ' ');
  }

  percentage(value: number): number {
    return Math.round(Math.min(Math.max(value, 0), 1) * 100);
  }

  primaryPlan(): AgentRetrievalPlan | null {
    if (this.result.retrieval_plan) {
      return this.result.retrieval_plan;
    }

    const executions = this.result.information_need_resolution?.executions ?? [];
    for (let index = executions.length - 1; index >= 0; index -= 1) {
      const execution = executions[index];
      if (execution.current_plan) {
        return execution.current_plan;
      }
      const lastPlan = execution.plan_history.at(-1);
      if (lastPlan) {
        return lastPlan;
      }
    }
    return null;
  }

  targetNeedCount(plan: AgentRetrievalPlan): number {
    return 'target_information_need_count' in plan ? plan.target_information_need_count : 1;
  }

  primaryStrategyLabel(): string {
    const plan = this.primaryPlan();
    return plan ? this.label(plan.strategy) : 'No plan recorded';
  }

  primaryPipelineName(): string | null {
    return this.primaryPlan()?.selected_pipeline_name ?? null;
  }


  primaryDocumentPreference(): DocumentPreference | null {
    if (this.result.primary_document_preference) {
      return this.result.primary_document_preference;
    }
    return this.primaryPlan()?.preferred_document ?? null;
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
    return (
      this.result.information_need_resolution?.total_retrieval_attempts ??
      this.result.retrieval_retry?.attempt_count ??
      (this.result.evidence.length > 0 ? 1 : 0)
    );
  }

  retryCount(): number {
    if (this.result.retrieval_retry) {
      return this.result.retrieval_retry.retries_used;
    }

    return (this.result.information_need_resolution?.executions ?? []).reduce(
      (total, execution) => total + Math.max(execution.attempts_used - 1, 0),
      0,
    );
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
