import { NgClass } from '@angular/common';
import { Component, Input } from '@angular/core';

import { BackgroundJob } from '../../models/background-job.models';

@Component({
  selector: 'app-document-job-progress',
  standalone: true,
  imports: [NgClass],
  templateUrl: './document-job-progress.component.html',
})
export class DocumentJobProgressComponent {
  @Input({ required: true }) job!: BackgroundJob;

  get percentage(): number {
    return Math.max(0, Math.min(100, Math.round(this.job.progress * 100)));
  }

  get title(): string {
    switch (this.job.job_type) {
      case 'ingest_document':
        return 'Processing uploaded document';
      case 'rebuild_document_index':
        return 'Rebuilding document index';
      case 'contextualize_document':
        return 'Contextualizing document';
      case 'delete_document':
        return 'Removing document';
      default:
        return 'Background work';
    }
  }

  get message(): string {
    if (this.job.status === 'failed') {
      return this.job.error_message || 'The background operation failed.';
    }
    if (this.job.status === 'cancelled') {
      return 'The background operation was cancelled.';
    }
    if (this.job.status === 'succeeded') {
      return this.job.job_type === 'delete_document'
        ? 'The document and its indexed data were removed.'
        : 'The background operation completed successfully.';
    }

    const stage = this.job.current_stage || this.job.status;
    if (stage === 'retry_scheduled') {
      return retryMessage(this.job.scheduled_at);
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
  queued: 'Waiting for an available worker.',
  claimed: 'A worker has claimed the job.',
  lease_expired: 'The previous worker stopped responding. This job is being recovered.',
  materializing_document: 'Downloading the stored source file for processing.',
  parsing_and_chunking: 'Extracting text, creating chunks, and generating original embeddings.',
  building_context_hierarchy: 'Building document and section-level context.',
  contextualization_complete: 'Chunk contextualization is complete.',
  indexing_document: 'Writing chunk and hierarchy vectors to Qdrant.',
  activating_document_version: 'Activating the completed document version.',
  replacing_chunk_registry: 'Replacing the stored chunk registry.',
  removing_superseded_points: 'Removing vectors that are no longer part of the rebuilt index.',
  loading_document_for_deletion: 'Loading document versions and storage references.',
  removing_document_vectors: 'Removing all document vectors from Qdrant.',
  removing_stored_source_files: 'Removing source files from object storage.',
  invalidating_keyword_index: 'Invalidating the derived keyword index.',
  removing_document_records: 'Removing document records from PostgreSQL.',
  document_deletion_complete: 'All document resources have been removed.',
  completed: 'The background operation completed successfully.',
  failed: 'The background operation failed.',
};

function humanizeStage(stage: string): string {
  const words = stage.replaceAll('_', ' ').trim();
  return words ? `${words.charAt(0).toUpperCase()}${words.slice(1)}.` : 'Background work is running.';
}

function retryMessage(scheduledAt: string): string {
  const retryAt = new Date(scheduledAt);
  if (Number.isNaN(retryAt.getTime())) {
    return 'The previous attempt failed. The worker will retry automatically.';
  }
  return `The previous attempt failed. The worker will retry at ${retryAt.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })}.`;
}
