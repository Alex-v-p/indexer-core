import { NgClass, NgFor, NgIf, PercentPipe } from '@angular/common';
import { Component, Input } from '@angular/core';

import {
  AttemptDocumentBalancing,
  DocumentPreference,
  EvidenceItem,
  InformationNeed,
  InformationNeedAttempt,
  InformationNeedDecomposition,
  InformationNeedResolution,
  PrimaryDocumentDetection,
  RerankingMetadata,
  RetrievalExecutionMetadata,
  TraceStep,
} from '../../models/query.models';
import { TraceEvidenceCardViewModel } from '../../view-models/trace-evidence-view.models';
import {
  buildAttemptEvidenceCard,
  evidencePreview,
  evidenceSource,
} from '../../utils/trace-evidence-view-model';
import { TraceEvidenceCardComponent } from '../trace-evidence-card/trace-evidence-card.component';

export interface PrimaryDocumentDetectionEvent {
  stepOrder: number;
  need: InformationNeed;
  detection: PrimaryDocumentDetection;
  preference: DocumentPreference | null;
  previousDocument: string | null;
  selectedDocument: string | null;
  inherited: boolean;
  supportingEvidence: TraceEvidenceCardViewModel[];
}

export interface BalancedRetrievalAttempt {
  need: InformationNeed;
  attempt: InformationNeedAttempt | null;
  stepOrder: number | null;
  attemptNumber: number;
  query: string;
  strategy: string;
  pipeline: string;
  planRationale: string | null;
  adjustments: string[];
  preferredDocument: DocumentPreference | null;
  retrieval: RetrievalExecutionMetadata;
  reranking: RerankingMetadata;
  balancing: AttemptDocumentBalancing;
  evidence: TraceEvidenceCardViewModel[];
}

@Component({
  selector: 'app-document-preference-trace',
  standalone: true,
  imports: [NgClass, NgFor, NgIf, PercentPipe, TraceEvidenceCardComponent],
  templateUrl: './document-preference-trace.component.html',
})
export class DocumentPreferenceTraceComponent {
  @Input() preference: DocumentPreference | null = null;
  @Input() decomposition: InformationNeedDecomposition | null = null;
  @Input() resolution: InformationNeedResolution | null = null;
  @Input() aggregateEvidence: EvidenceItem[] = [];
  @Input() trace: TraceStep[] = [];

  detectionEvents(): PrimaryDocumentDetectionEvent[] {
    return this.trace.flatMap((step) => {
      const detection = this.primaryDocumentDetection(step.metadata['primary_document_detection']);
      if (!detection) {
        return [];
      }
      const preference = this.documentPreference(step.metadata['primary_document_preference']);
      const need = this.needById(detection.information_need_id);
      return [{
        stepOrder: step.step_order,
        need,
        detection,
        preference,
        previousDocument: this.documentLabel(detection.previous_document_key),
        selectedDocument:
          preference?.document.display_name ?? this.documentLabel(detection.selected_document_key),
        inherited: detection.previous_document_key !== null,
        supportingEvidence: (preference?.supporting_evidence_ranks ?? [])
          .map((rank) => this.supportingEvidenceCard(rank))
          .filter((item): item is TraceEvidenceCardViewModel => item !== null),
      }];
    });
  }

  balancingEvents(): BalancedRetrievalAttempt[] {
    const structured = (this.resolution?.executions ?? []).flatMap((execution) =>
      [...(execution.attempts ?? [])]
        .sort((left, right) => left.attempt_number - right.attempt_number)
        .map((attempt) => this.structuredAttempt(execution.information_need, attempt)),
    );
    if (structured.length > 0) {
      return structured;
    }

    return this.legacyBalancingEvents();
  }

  selectedDocumentCounts(
    event: BalancedRetrievalAttempt,
  ): Array<{ key: string; count: number }> {
    return Object.entries(event.balancing.selected_chunks_per_document ?? {})
      .map(([key, count]) => ({ key, count }))
      .sort((left, right) => right.count - left.count);
  }

  requestedTopK(event: BalancedRetrievalAttempt): number | null {
    return (
      event.retrieval.requested_top_k ??
      event.balancing.requested_top_k ??
      event.attempt?.top_k ??
      null
    );
  }

  retrievalMethod(event: BalancedRetrievalAttempt): string {
    return this.label(event.retrieval.retriever_type ?? event.strategy);
  }

  retrievalTechnique(event: BalancedRetrievalAttempt): string | null {
    if (event.retrieval.fusion_method) {
      const contributions = [
        event.retrieval.vector_contribution === null
          ? null
          : `vector ${event.retrieval.vector_contribution}`,
        event.retrieval.keyword_contribution === null
          ? null
          : `keyword ${event.retrieval.keyword_contribution}`,
      ].filter(Boolean);
      return [
        `${this.label(event.retrieval.fusion_method)} fusion`,
        contributions.length > 0 ? contributions.join(' · ') : null,
      ].filter(Boolean).join(' · ');
    }
    if (
      event.retrieval.hierarchical_document_candidate_count !== null ||
      event.retrieval.hierarchical_section_candidate_count !== null
    ) {
      return [
        'hierarchical',
        event.retrieval.hierarchical_document_candidate_count === null
          ? null
          : `${event.retrieval.hierarchical_document_candidate_count} document candidates`,
        event.retrieval.hierarchical_section_candidate_count === null
          ? null
          : `${event.retrieval.hierarchical_section_candidate_count} section candidates`,
      ].filter(Boolean).join(' · ');
    }
    if (event.retrieval.multi_query_query_count !== null) {
      return `multi-query fusion · ${event.retrieval.multi_query_query_count} query variants`;
    }
    return null;
  }

  rerankingLabel(event: BalancedRetrievalAttempt): string {
    if (!event.reranking.applied) {
      return 'Reranking not applied';
    }
    const counts =
      event.reranking.candidate_count_before === null
        ? null
        : `${event.reranking.candidate_count_before} → ${
            event.reranking.candidate_count_after ?? event.balancing.selected_count ?? '?'
          } candidates`;
    return [
      `Reranked${event.reranking.provider ? ` by ${event.reranking.provider}` : ''}`,
      counts,
    ].filter(Boolean).join(' · ');
  }

  documentKeyLabel(value: string): string {
    const label = value.replace(/^(document|version|name):/, '');
    return label.length > 36 ? `${label.slice(0, 33)}…` : label;
  }

  label(value: string): string {
    return value.replaceAll('_', ' ');
  }

  trackDetection(index: number, event: PrimaryDocumentDetectionEvent): string {
    return `${event.stepOrder}-${event.need.need_id}-${index}`;
  }

  trackAttempt(index: number, event: BalancedRetrievalAttempt): string {
    return `${event.need.need_id}-${event.attemptNumber}-${index}`;
  }

  trackDocument(index: number, item: { key: string }): string {
    return item.key || String(index);
  }

  trackEvidence(index: number, item: TraceEvidenceCardViewModel): string {
    return item.key || String(index);
  }

  private structuredAttempt(
    need: InformationNeed,
    attempt: InformationNeedAttempt,
  ): BalancedRetrievalAttempt {
    const preference = attempt.plan?.preferred_document ?? null;
    const aggregateKeys = this.aggregateEvidenceKeys();
    return {
      need,
      attempt,
      stepOrder: null,
      attemptNumber: attempt.attempt_number,
      query: attempt.query || attempt.plan?.query || need.retrieval_query,
      strategy: attempt.strategy || attempt.plan?.strategy || 'unknown',
      pipeline: attempt.pipeline_name || attempt.plan?.selected_pipeline_name || 'unknown',
      planRationale: attempt.plan?.rationale ?? null,
      adjustments: attempt.adjustments ?? attempt.plan?.adjustments ?? [],
      preferredDocument: preference,
      retrieval: attempt.retrieval_metadata ?? this.emptyRetrievalMetadata(),
      reranking: attempt.reranking_metadata ?? this.emptyRerankingMetadata(),
      balancing: attempt.document_balancing ?? this.emptyBalancingMetadata(),
      evidence: (attempt.evidence ?? []).map((item) =>
        buildAttemptEvidenceCard(item, {
          informationNeeds: this.informationNeeds(),
          preferredDocument: preference,
          retainedInAggregate:
            aggregateKeys.has(item.evidence_key) ||
            this.aggregateEvidence.some((candidate) => candidate.rank === item.aggregate_rank),
        }),
      ),
    };
  }

  private legacyBalancingEvents(): BalancedRetrievalAttempt[] {
    return this.trace.flatMap((step) => {
      const lookup = this.record(step.metadata['active_information_need_lookup']);
      if (!lookup) {
        return [];
      }
      const balancing = this.legacyDocumentBalancing(lookup['document_balancing']);
      if (!balancing) {
        return [];
      }
      const needId = this.string(lookup['information_need_id']) ?? 'unknown-need';
      const need = this.needById(needId);
      return [{
        need,
        attempt: null,
        stepOrder: step.step_order,
        attemptNumber: this.number(lookup['attempt_number']) ?? 0,
        query: need.retrieval_query,
        strategy: 'unknown',
        pipeline: 'unknown',
        planRationale: null,
        adjustments: [],
        preferredDocument: null,
        retrieval: this.emptyRetrievalMetadata(),
        reranking: this.emptyRerankingMetadata(),
        balancing,
        evidence: [],
      }];
    });
  }

  private supportingEvidenceCard(rank: number): TraceEvidenceCardViewModel | null {
    for (const execution of this.resolution?.executions ?? []) {
      for (const attempt of execution.attempts ?? []) {
        const item = (attempt.evidence ?? []).find((candidate) => candidate.aggregate_rank === rank);
        if (item) {
          return buildAttemptEvidenceCard(item, {
            informationNeeds: this.informationNeeds(),
            retainedInAggregate: this.aggregateEvidenceKeys().has(item.evidence_key),
          });
        }
      }
    }
    const aggregate = this.aggregateEvidence.find((item) => item.rank === rank);
    if (!aggregate) {
      return null;
    }
    return {
      key: aggregate.id ?? aggregate.qdrant_chunk_index_id ?? `aggregate-rank:${rank}`,
      text: aggregate.text,
      preview: evidencePreview(aggregate.text),
      aggregateRank: aggregate.rank,
      retrievalOrder: null,
      retrievalScore: aggregate.score,
      source: evidenceSource(aggregate),
      preferredDocument: null,
      attemptGrade: null,
      finalGrade: null,
      supportedNeeds: [],
      retainedAfterNeedGrading: null,
      retainedInAggregate: true,
      usedByAnswer: null,
      citations: [],
    };
  }

  private informationNeeds(): InformationNeed[] {
    const lookup = new Map<string, InformationNeed>();
    for (const need of this.decomposition?.information_needs ?? []) {
      lookup.set(need.need_id, need);
    }
    for (const execution of this.resolution?.executions ?? []) {
      lookup.set(execution.information_need_id, execution.information_need);
    }
    return [...lookup.values()];
  }

  private needById(needId: string): InformationNeed {
    return this.informationNeeds().find((need) => need.need_id === needId) ?? {
      need_id: needId,
      description: 'Information need details unavailable',
      retrieval_query: 'Retrieval query unavailable for this older run',
      required: false,
    };
  }

  private aggregateEvidenceKeys(): Set<string> {
    return new Set(
      (this.resolution?.executions ?? []).flatMap((execution) => execution.evidence_keys ?? []),
    );
  }

  private documentLabel(key: string | null): string | null {
    if (!key) {
      return null;
    }
    if (this.preference?.document.key === key) {
      return this.preference.document.display_name;
    }
    for (const step of this.trace) {
      const candidate = this.documentPreference(step.metadata['primary_document_preference']);
      if (candidate?.document.key === key) {
        return candidate.document.display_name;
      }
    }
    return this.documentKeyLabel(key);
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

  private legacyDocumentBalancing(value: unknown): AttemptDocumentBalancing | null {
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
      candidate_count: this.number(record['candidate_count']),
      requested_top_k: this.number(record['requested_top_k']),
      selected_count: this.number(record['selected_count']),
      selected_chunks_per_document: normalizedCounts,
      primary_document_key: this.string(record['primary_document_key']),
      primary_document_quota: this.number(record['primary_document_quota']),
      primary_document_selected_count: this.number(record['primary_selected_count']),
      quota_relaxed: this.boolean(record['quota_relaxed']) ?? false,
      preferred_document_active: this.string(record['primary_document_key']) !== null,
      details: {},
    };
  }

  private emptyRetrievalMetadata(): RetrievalExecutionMetadata {
    return {
      requested_top_k: null,
      candidate_top_k: null,
      retriever_type: null,
      fusion_method: null,
      vector_contribution: null,
      keyword_contribution: null,
      query_variants: [],
      multi_query_query_count: null,
      multi_query_candidate_top_k_per_query: null,
      multi_query_result_counts: {},
      hierarchical_document_candidate_count: null,
      hierarchical_selected_document_version_ids: [],
      hierarchical_section_candidate_count: null,
      hierarchical_selected_section_ids: [],
      details: {},
    };
  }

  private emptyRerankingMetadata(): RerankingMetadata {
    return {
      applied: false,
      provider: null,
      candidate_count_before: null,
      candidate_count_after: null,
      details: {},
    };
  }

  private emptyBalancingMetadata(): AttemptDocumentBalancing {
    return {
      selector_name: null,
      candidate_count: null,
      requested_top_k: null,
      selected_count: null,
      selected_chunks_per_document: {},
      primary_document_key: null,
      primary_document_quota: null,
      primary_document_selected_count: null,
      quota_relaxed: false,
      preferred_document_active: false,
      details: {},
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
