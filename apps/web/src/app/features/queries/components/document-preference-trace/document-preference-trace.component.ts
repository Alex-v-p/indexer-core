import { NgClass, NgFor, NgIf, PercentPipe } from '@angular/common';
import { Component, Input } from '@angular/core';

import {
  DocumentBalancing,
  DocumentPreference,
  PrimaryDocumentDetection,
  TraceStep,
} from '../../models/query.models';

interface PrimaryDocumentDetectionEvent {
  stepOrder: number;
  detection: PrimaryDocumentDetection;
  preference: DocumentPreference | null;
}

interface DocumentBalancingEvent {
  stepOrder: number;
  informationNeedId: string;
  attemptNumber: number;
  primaryDocument: string | null;
  balancing: DocumentBalancing;
}

@Component({
  selector: 'app-document-preference-trace',
  standalone: true,
  imports: [NgClass, NgFor, NgIf, PercentPipe],
  templateUrl: './document-preference-trace.component.html',
})
export class DocumentPreferenceTraceComponent {
  @Input() preference: DocumentPreference | null = null;
  @Input() trace: TraceStep[] = [];

  detectionEvents(): PrimaryDocumentDetectionEvent[] {
    return this.trace.flatMap((step) => {
      const detection = this.primaryDocumentDetection(step.metadata['primary_document_detection']);
      if (!detection) {
        return [];
      }
      return [{
        stepOrder: step.step_order,
        detection,
        preference: this.documentPreference(step.metadata['primary_document_preference']),
      }];
    });
  }

  balancingEvents(): DocumentBalancingEvent[] {
    return this.trace.flatMap((step) => {
      const lookup = this.record(step.metadata['active_information_need_lookup']);
      if (!lookup) {
        return [];
      }
      const balancing = this.documentBalancing(lookup['document_balancing']);
      if (!balancing) {
        return [];
      }
      return [{
        stepOrder: step.step_order,
        informationNeedId: this.string(lookup['information_need_id']) ?? 'unknown need',
        attemptNumber: this.number(lookup['attempt_number']) ?? 0,
        primaryDocument: this.string(lookup['primary_document']),
        balancing,
      }];
    });
  }

  selectedDocumentCounts(event: DocumentBalancingEvent): Array<{ key: string; count: number }> {
    return Object.entries(event.balancing.selected_document_counts).map(([key, count]) => ({ key, count }));
  }

  documentKeyLabel(value: string): string {
    const label = value.replace(/^(document|version|name):/, '');
    return label.length > 28 ? `${label.slice(0, 25)}…` : label;
  }

  private primaryDocumentDetection(value: unknown): PrimaryDocumentDetection | null {
    const record = this.record(value);
    const informationNeedId = this.string(record?.['information_need_id']);
    const detectorName = this.string(record?.['detector_name']);
    if (!record || !informationNeedId || !detectorName) {
      return null;
    }
    return {
      information_need_id: informationNeedId,
      detector_name: detectorName,
      candidate_evidence_count: this.number(record['candidate_evidence_count']) ?? 0,
      previous_document_key: this.string(record['previous_document_key']),
      selected_document_key: this.string(record['selected_document_key']),
      preference_changed: this.boolean(record['preference_changed']) ?? false,
    };
  }

  private documentPreference(value: unknown): DocumentPreference | null {
    const record = this.record(value);
    const document = this.record(record?.['document']);
    const key = this.string(document?.['key']);
    const displayName = this.string(document?.['display_name']);
    if (!record || !document || !key || !displayName) {
      return null;
    }
    return {
      document: {
        key,
        display_name: displayName,
        document_id: this.string(document['document_id']),
        document_version_ids: this.stringArray(document['document_version_ids']),
        normalized_names: this.stringArray(document['normalized_names']),
      },
      score: this.number(record['score']) ?? 0,
      confidence: this.number(record['confidence']) ?? 0,
      margin: this.number(record['margin']) ?? 0,
      supporting_information_need_ids: this.stringArray(record['supporting_information_need_ids']),
      supporting_evidence_ranks: this.numberArray(record['supporting_evidence_ranks']),
      rationale: this.string(record['rationale']) ?? 'No rationale was recorded.',
      detector_name: this.string(record['detector_name']) ?? 'unknown detector',
      semantics: this.string(record['semantics']) ?? 'soft_preference_not_filter',
    };
  }

  private documentBalancing(value: unknown): DocumentBalancing | null {
    const record = this.record(value);
    const selectorName = this.string(record?.['selector_name']);
    if (!record || !selectorName) {
      return null;
    }
    const selectedCounts = this.record(record['selected_document_counts']) ?? {};
    const normalizedCounts: Record<string, number> = {};
    for (const [key, count] of Object.entries(selectedCounts)) {
      const parsed = this.number(count);
      if (parsed !== null) {
        normalizedCounts[key] = parsed;
      }
    }
    return {
      selector_name: selectorName,
      candidate_count: this.number(record['candidate_count']) ?? 0,
      requested_top_k: this.number(record['requested_top_k']) ?? 0,
      selected_count: this.number(record['selected_count']) ?? 0,
      selected_document_counts: normalizedCounts,
      primary_document_key: this.string(record['primary_document_key']),
      primary_selected_count: this.number(record['primary_selected_count']) ?? 0,
      quota_relaxed: this.boolean(record['quota_relaxed']) ?? false,
    };
  }

  private record(value: unknown): Record<string, unknown> | null {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  }

  private string(value: unknown): string | null {
    return typeof value === 'string' && value.trim().length > 0 ? value : null;
  }

  private number(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  private boolean(value: unknown): boolean | null {
    return typeof value === 'boolean' ? value : null;
  }

  private stringArray(value: unknown): string[] {
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
  }

  private numberArray(value: unknown): number[] {
    return Array.isArray(value)
      ? value.filter((item): item is number => typeof item === 'number' && Number.isFinite(item))
      : [];
  }
}
