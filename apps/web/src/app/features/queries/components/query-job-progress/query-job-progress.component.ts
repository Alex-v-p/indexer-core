import { NgClass } from '@angular/common';
import { Component, Input } from '@angular/core';

import { BackgroundJob } from '../../../../core/background-jobs/background-job.models';

@Component({
  selector: 'app-query-job-progress',
  standalone: true,
  imports: [NgClass],
  templateUrl: './query-job-progress.component.html',
})
export class QueryJobProgressComponent {
  @Input({ required: true }) job!: BackgroundJob;

  get percentage(): number {
    return Math.max(0, Math.min(100, Math.round(this.job.progress * 100)));
  }

  get message(): string {
    if (this.job.status === 'failed') {
      return this.job.error_message || 'The query could not be completed.';
    }
    if (this.job.status === 'cancelled') {
      return 'The query job was cancelled.';
    }
    if (this.job.status === 'succeeded') {
      return 'The answer is ready. Loading the persisted query result…';
    }

    const stage = this.job.current_stage || this.job.status;
    if (stage === 'retry_scheduled') {
      return retryMessage(this.job.scheduled_at);
    }
    if (stage === 'queued' && isFuture(this.job.scheduled_at)) {
      return scheduledMessage(this.job.scheduled_at);
    }
    return STAGE_MESSAGES[stage] || humanizeStage(stage);
  }

  get statusClasses(): string {
    switch (this.job.status) {
      case 'succeeded':
        return 'border-success/30 bg-success/10';
      case 'failed':
        return 'border-danger/30 bg-danger/10';
      case 'cancelled':
        return 'border-border-strong bg-surface-muted';
      default:
        return 'border-primary/30 bg-primary-soft';
    }
  }
}

const STAGE_MESSAGES: Record<string, string> = {
  queued: 'Waiting for an available query worker.',
  claimed: 'A worker has claimed the complete query run.',
  lease_expired: 'The previous worker stopped responding. Another worker is recovering this query.',
  preparing_query_execution: 'Preparing the selected query pipeline.',
  classifying_query: 'Classifying the question and its retrieval requirements.',
  decomposing_information_needs: 'Breaking the question into the information that must be resolved.',
  preparing_information_needs: 'Preparing the information-need work queue.',
  resolving_information_needs: 'Resolving the required information needs.',
  selecting_information_need: 'Selecting the next unresolved information need.',
  classifying_information_need: 'Classifying the current information need.',
  planning_retrieval: 'Choosing the retrieval strategy for the current information need.',
  retrieving_evidence: 'Searching the indexed documents for supporting evidence.',
  reranking_evidence: 'Reranking the strongest evidence candidates.',
  validating_evidence_constraints: 'Checking document, version, and date constraints.',
  grading_evidence: 'Grading whether the retrieved evidence is sufficient.',
  deciding_retrieval_retry: 'Deciding whether another retrieval attempt is needed.',
  selecting_primary_document: 'Determining the primary supporting document.',
  completing_information_need: 'Finalizing the current information need.',
  aggregating_information_needs: 'Combining the resolved information needs.',
  arbitrating_final_evidence: 'Selecting the final evidence that may support the answer.',
  preparing_answer_context: 'Preparing the grounded context for answer generation.',
  generating_answer: 'Generating the final grounded answer and citations.',
  saving_query_result: 'Saving the answer, citations, evidence, and execution trace.',
  query_result_already_available: 'The query result was already completed by an earlier worker attempt.',
  running_query_graph: 'Running the query graph.',
  completed: 'The query completed successfully.',
  failed: 'The query failed.',
};

function humanizeStage(stage: string): string {
  const normalized = stage.endsWith('_failed') ? stage.slice(0, -7) : stage;
  const words = normalized.replaceAll('_', ' ').trim();
  const message = words ? `${words.charAt(0).toUpperCase()}${words.slice(1)}.` : 'The query is running.';
  return stage.endsWith('_failed') ? `${message} The worker will apply the configured retry policy.` : message;
}

function isFuture(value: string): boolean {
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) && timestamp > Date.now() + 1_000;
}

function scheduledMessage(value: string): string {
  const scheduledAt = new Date(value);
  return `This query is scheduled for ${scheduledAt.toLocaleString()}.`;
}

function retryMessage(value: string): string {
  const retryAt = new Date(value);
  if (Number.isNaN(retryAt.getTime())) {
    return 'The previous attempt failed. The worker will retry automatically.';
  }
  return `The previous attempt failed. A worker may retry at ${retryAt.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })}.`;
}
