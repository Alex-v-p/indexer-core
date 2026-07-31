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
      case 'delete_document_versions':
        return this.deletionTargetCount > 1
          ? `Removing ${this.deletionTargetCount} document versions`
          : 'Removing document version';
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
      return this.job.job_type === 'delete_document_versions'
        ? this.deletionTargetCount > 1
          ? 'The selected document versions were removed.'
          : 'The selected document version was removed.'
        : 'The background operation completed successfully.';
    }

    const stage = this.job.current_stage || this.job.status;
    if (stage === 'retry_scheduled') {
      return retryMessage(this.job.scheduled_at);
    }
    return STAGE_MESSAGES[stage] || humanizeStage(stage);
  }


  private get deletionTargetCount(): number {
    const targets = this.job.payload['targets'];
    return Array.isArray(targets) ? targets.length : 1;
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
  loading_document_version_for_deletion: 'Loading the selected document version and its storage reference.',
  removing_version_vectors: 'Removing only the selected version vectors from Qdrant.',
  removing_version_source_file: 'Removing the selected version source file from object storage.',
  updating_document_version_records: 'Removing the selected version records from PostgreSQL.',
  promoting_remaining_document_version: 'Promoting the newest remaining ready version.',
  invalidating_keyword_index: 'Invalidating the derived keyword index.',
  document_version_deletion_complete: 'All selected document versions have been removed.',
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
