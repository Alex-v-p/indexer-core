import { TestBed } from '@angular/core/testing';

import {
  DocumentPreference,
  EvidenceGrading,
  InformationNeedAttempt,
  InformationNeedRetrievalPlan,
  QueryResponse,
} from '../../models/query.models';
import { formatMetadataValue } from '../../utils/agent-trace-view-model';
import { AgentJourneyOverviewComponent } from './agent-journey-overview.component';

describe('AgentJourneyOverviewComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AgentJourneyOverviewComponent],
    }).compileComponents();
  });

  it('renders separate need lanes and ordered retry transitions', () => {
    const fixture = TestBed.createComponent(AgentJourneyOverviewComponent);
    fixture.componentInstance.result = buildQueryResponse();
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    const lanes = element.querySelectorAll('[data-testid="need-lane"]');
    const attempts = Array.from(
      element.querySelectorAll('[data-testid="journey-attempt"]'),
    ).map(
      (node) => node.textContent ?? '',
    );
    const transition = element.querySelector('[data-testid="retry-transition"]');
    const text = element.textContent ?? '';

    expect(lanes).toHaveLength(2);
    expect(attempts).toHaveLength(3);
    expect(attempts[0]).toContain('Attempt 1');
    expect(attempts[0]).toContain('hybrid-rag');
    expect(attempts[0]).toContain('hybrid');
    expect(attempts[1]).toContain('Attempt 2');
    expect(attempts[1]).toContain('multi-query-rag');
    expect(attempts[1]).toContain('multi query');
    expect(transition?.textContent).toContain('weak evidence');
    expect(transition?.textContent).toContain('broaden query');
    expect(transition?.textContent).toContain('switched hybrid to multi query');
    expect(text).not.toContain('Retrieval strategy evolution');
    expect(text).not.toContain('Initial query-level plan');
    expect(text).not.toContain('Selected retrieval plan');
  });

  it('derives strategy, pipeline, retry, preference, and evidence lookups', () => {
    const component = new AgentJourneyOverviewComponent();
    component.result = buildQueryResponse();

    const viewModel = component.viewModel;
    expect(viewModel?.summary.strategiesUsed).toEqual([
      'hybrid',
      'multi_query',
      'hierarchical',
    ]);
    expect(viewModel?.summary.pipelinesUsed).toEqual([
      'hybrid-rag',
      'multi-query-rag',
      'hierarchical-rag',
    ]);
    expect(viewModel?.summary.strategyChanges).toBe(1);
    expect(viewModel?.summary.pipelineChanges).toBe(1);
    expect(viewModel?.summary.totalAttempts).toBe(3);
    expect(viewModel?.summary.retries).toBe(1);
    expect(viewModel?.needLanes[0].attempts[1].preferredDocument?.document.display_name).toBe(
      'deployment-guide.pdf',
    );
    expect(viewModel?.informationNeedsById.get('need-2')?.description).toBe(
      'Explain deployment architecture',
    );
    expect(viewModel?.evidenceByRank.get(2)?.text).toBe('Rejected but inspectable evidence.');
    expect(viewModel?.evidenceByKey.get('qdrant_chunk:chunk-2')?.text).toBe(
      'Rejected but inspectable evidence.',
    );
    expect(viewModel?.sourceLabelsByEvidenceKey.get('qdrant_chunk:chunk-2')).toBe(
      'deployment-guide.pdf',
    );
    expect(formatMetadataValue(['page 3', 4, null])).toBe('page 3, 4');
  });

  it('renders safely when optional structured trace data is missing', () => {
    const fixture = TestBed.createComponent(AgentJourneyOverviewComponent);
    const olderResult = buildQueryResponse();
    olderResult.classification = null;
    olderResult.retrieval_plan = null;
    olderResult.information_need_resolution = null;
    olderResult.primary_document_preference = null;
    olderResult.evidence_grading = null;
    olderResult.trace = [];

    expect(() => {
      fixture.componentInstance.result = olderResult;
      fixture.detectChanges();
    }).not.toThrow();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelectorAll('[data-testid="need-lane"]')).toHaveLength(2);
    expect(element.querySelectorAll('[data-testid="journey-attempt"]')).toHaveLength(0);
    expect(element.textContent).toContain('No structured attempts were recorded');
  });
});

function buildQueryResponse(): QueryResponse {
  const firstPlan = plan({
    needId: 'need-1',
    attemptNumber: 1,
    strategy: 'hybrid',
    pipeline: 'hybrid-rag',
    query: 'deployment requirements',
    topK: 5,
    rationale: 'Combine vector and keyword retrieval.',
  });
  const retryPlan = plan({
    needId: 'need-1',
    attemptNumber: 2,
    strategy: 'multi_query',
    pipeline: 'multi-query-rag',
    query: 'production deployment requirements and constraints',
    topK: 10,
    rationale: 'Broaden coverage with query variants.',
    adjustments: ['broaden_query', 'increase_top_k'],
    preferredDocument: preference(),
  });
  const secondNeedPlan = plan({
    needId: 'need-2',
    attemptNumber: 1,
    strategy: 'hierarchical',
    pipeline: 'hierarchical-rag',
    query: 'deployment architecture sections',
    topK: 6,
    rationale: 'Navigate document and section summaries.',
  });

  const weakAttempt = attempt({
    plan: firstPlan,
    status: 'weak',
    coverage: 0.35,
    rationale: 'The evidence only partially covered deployment constraints.',
    retrievedCount: 5,
    uniqueCount: 4,
  });
  const successfulRetry = attempt({
    plan: retryPlan,
    status: 'sufficient',
    coverage: 0.92,
    rationale: 'The broadened retrieval directly covers the need.',
    retrievedCount: 8,
    uniqueCount: 3,
    evidence: [
      {
        evidence_key: 'qdrant_chunk:chunk-2',
        retrieval_order: 2,
        aggregate_rank: 2,
        text: 'Rejected but inspectable evidence.',
        score: 0.42,
        qdrant_chunk_index_id: 'chunk-2',
        document_id: 'document-1',
        document_version_id: 'version-1',
        metadata: { original_filename: 'deployment-guide.pdf' },
        relevance_score: 0.2,
        relevant: false,
        grading_rationale: 'This passage does not answer the need.',
        supports_information_need_ids: [],
        retained_after_need_grading: false,
      },
    ],
  });
  const hierarchicalAttempt = attempt({
    plan: secondNeedPlan,
    status: 'sufficient',
    coverage: 0.84,
    rationale: 'The architecture sections resolve the need.',
    retrievedCount: 6,
    uniqueCount: 5,
  });

  return {
    id: 'run-2',
    question: 'Explain the deployment requirements and architecture.',
    answer: 'The application uses containerized services.',
    status: 'succeeded',
    pipeline_name: 'agentic-rag',
    pipeline_version: '1.0.0',
    top_k: 5,
    started_at: null,
    completed_at: null,
    error_message: null,
    classification: classification(),
    information_need_decomposition: {
      information_needs: [
        {
          need_id: 'need-1',
          description: 'Identify deployment requirements',
          retrieval_query: 'deployment requirements',
          required: true,
        },
        {
          need_id: 'need-2',
          description: 'Explain deployment architecture',
          retrieval_query: 'deployment architecture sections',
          required: false,
        },
      ],
      information_need_count: 2,
      rationale: 'Requirements and architecture are separate needs.',
      decomposer_name: 'test',
      fallback_used: false,
    },
    retrieval_plan: {
      strategy: 'baseline',
      selected_pipeline_name: 'baseline-rag',
      rationale: 'Start with a simple query-level lookup.',
      planner_name: 'test',
      based_on_query_type: 'broad_explanation',
      metadata_filter_hints: [],
      requires_reranking: false,
      target_information_need_ids: ['need-1', 'need-2'],
      target_information_need_count: 2,
      document_constraint: documentConstraint(),
      version_constraint: versionConstraint(),
      date_constraints: [],
      preferred_document: null,
    },
    evidence_grading: null,
    retrieval_retry: null,
    primary_document_preference: preference(),
    information_need_resolution: {
      graph_name: 'information_need_resolution',
      information_need_count: 2,
      supported_information_need_ids: ['need-1', 'need-2'],
      supported_information_need_count: 2,
      unresolved_information_need_ids: [],
      unresolved_information_need_count: 0,
      complete: true,
      total_retrieval_attempts: 3,
      max_total_retrieval_attempts: 8,
      max_attempts_per_information_need: 3,
      executions: [
        {
          information_need: {
            need_id: 'need-1',
            description: 'Identify deployment requirements',
            retrieval_query: 'deployment requirements',
            required: true,
          },
          information_need_id: 'need-1',
          status: 'supported',
          attempts_used: 2,
          max_attempts: 3,
          reclassifications_used: 0,
          parent_information_need_id: null,
          depth: 0,
          classification: classification(),
          classification_history: [classification()],
          current_plan: retryPlan,
          plan_history: [firstPlan, retryPlan],
          attempts: [successfulRetry, weakAttempt],
          constraint_validation_history: [],
          evidence_keys: ['qdrant_chunk:chunk-1'],
          final_grade: {
            information_need_id: 'need-1',
            description: 'Identify deployment requirements',
            status: 'supported',
            coverage_score: 0.92,
            supporting_evidence_ranks: [1],
            rationale: 'The retry resolved the information need.',
            required: true,
          },
          stop_reason: 'supported',
          stop_rationale: 'Evidence coverage reached the support threshold.',
        },
        {
          information_need: {
            need_id: 'need-2',
            description: 'Explain deployment architecture',
            retrieval_query: 'deployment architecture sections',
            required: false,
          },
          information_need_id: 'need-2',
          status: 'supported',
          attempts_used: 1,
          max_attempts: 3,
          reclassifications_used: 0,
          parent_information_need_id: null,
          depth: 0,
          classification: classification(),
          classification_history: [classification()],
          current_plan: secondNeedPlan,
          plan_history: [secondNeedPlan],
          attempts: [hierarchicalAttempt],
          constraint_validation_history: [],
          evidence_keys: ['qdrant_chunk:chunk-3'],
          final_grade: {
            information_need_id: 'need-2',
            description: 'Explain deployment architecture',
            status: 'supported',
            coverage_score: 0.84,
            supporting_evidence_ranks: [3],
            rationale: 'The hierarchical lookup resolved the need.',
            required: false,
          },
          stop_reason: 'supported',
          stop_rationale: 'Evidence coverage reached the support threshold.',
        },
      ],
    },
    constraint_validation: null,
    evidence_context: null,
    evidence: [
      {
        id: 'evidence-1',
        rank: 1,
        score: 0.94,
        text: 'Accepted deployment evidence.',
        qdrant_chunk_index_id: 'chunk-1',
        document_id: 'document-1',
        document_version_id: 'version-1',
        metadata: { original_filename: 'deployment-guide.pdf' },
      },
    ],
    citations: [],
    trace: [
      {
        id: null,
        step_order: 20,
        name: 'arbitrate_final_evidence',
        step_type: 'evidence_arbitration',
        status: 'completed',
        duration_ms: 3,
        input_summary: null,
        output_summary: null,
        error_message: null,
        metadata: {},
      },
    ],
  };
}

function plan(options: {
  needId: string;
  attemptNumber: number;
  strategy: InformationNeedRetrievalPlan['strategy'];
  pipeline: string;
  query: string;
  topK: number;
  rationale: string;
  adjustments?: string[];
  preferredDocument?: InformationNeedRetrievalPlan['preferred_document'];
}): InformationNeedRetrievalPlan {
  return {
    information_need_id: options.needId,
    strategy: options.strategy,
    selected_pipeline_name: options.pipeline,
    query: options.query,
    top_k: options.topK,
    rationale: options.rationale,
    planner_name: 'test',
    based_on_query_type: 'broad_explanation',
    attempt_number: options.attemptNumber,
    metadata_filter_hints: [],
    requires_reranking: options.strategy === 'rerank',
    adjustments: options.adjustments ?? [],
    document_constraint: documentConstraint(),
    version_constraint: versionConstraint(),
    date_constraints: [],
    preferred_document: options.preferredDocument ?? null,
  };
}

function attempt(options: {
  plan: InformationNeedRetrievalPlan;
  status: EvidenceGrading['status'];
  coverage: number;
  rationale: string;
  retrievedCount: number;
  uniqueCount: number;
  evidence?: InformationNeedAttempt['evidence'];
}): InformationNeedAttempt {
  const grading = gradingReport(options.status, options.coverage, options.rationale);
  return {
    attempt_number: options.plan.attempt_number,
    query: options.plan.query,
    top_k: options.plan.top_k,
    pipeline_name: options.plan.selected_pipeline_name,
    strategy: options.plan.strategy,
    adjustments: options.plan.adjustments,
    retrieved_count: options.retrievedCount,
    unique_evidence_added: options.uniqueCount,
    evidence_keys: [],
    evidence: options.evidence ?? [],
    retrieval_metadata: {
      requested_top_k: options.plan.top_k,
      candidate_top_k: options.plan.top_k,
      retriever_type: options.plan.strategy,
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
    },
    reranking_metadata: {
      applied: false,
      provider: null,
      candidate_count_before: null,
      candidate_count_after: null,
      details: {},
    },
    document_balancing: {
      selector_name: null,
      candidate_count: null,
      requested_top_k: options.plan.top_k,
      selected_count: options.retrievedCount,
      selected_chunks_per_document: {},
      primary_document_key: null,
      primary_document_quota: null,
      primary_document_selected_count: null,
      quota_relaxed: false,
      preferred_document_active: options.plan.preferred_document !== null,
      details: {},
    },
    plan: options.plan,
    constraint_validation: {
      status: 'not_requested',
      blocked: false,
      candidate_count: options.retrievedCount,
      matched_count: options.retrievedCount,
      rejected_count: 0,
      constraints: {},
      rationale: 'No constraints requested.',
    },
    evidence_grading: grading,
  };
}

function gradingReport(
  status: EvidenceGrading['status'],
  coverage: number,
  rationale: string,
): EvidenceGrading {
  return {
    status,
    coverage_score: coverage,
    sufficient: status === 'sufficient',
    answerable: status === 'sufficient',
    partial_answer_available: status === 'weak',
    missing_evidence: status === 'missing',
    weak_evidence: status === 'weak',
    relevant_count: status === 'missing' ? 0 : 1,
    total_count: 1,
    relevant_evidence_ranks: status === 'missing' ? [] : [1],
    supported_information_need_count: status === 'sufficient' ? 1 : 0,
    supported_required_information_need_count: status === 'sufficient' ? 1 : 0,
    required_information_need_count: 1,
    supported_information: [],
    partial_information_need_count: status === 'weak' ? 1 : 0,
    missing_information_need_count: status === 'missing' ? 1 : 0,
    total_information_need_count: 1,
    unresolved_information: [],
    rationale,
    grader_name: 'test',
    fallback_used: false,
    grades: [],
    information_need_grades: [],
  };
}

function classification() {
  return {
    query_type: 'broad_explanation' as const,
    confidence: 0.9,
    needs_metadata_filters: false,
    metadata_filter_hints: [],
    rationale: 'The query requests a broad explanation.',
    classifier_name: 'test',
    fallback_used: false,
    document_constraint: documentConstraint(),
    version_constraint: versionConstraint(),
    date_constraints: [],
  };
}

function documentConstraint() {
  return {
    names: [],
    normalized_names: [],
    active: false,
    confidence: 0,
    rationale: 'No document constraint.',
    detector_name: 'none',
    match_semantics: 'exact_normalized_any' as const,
  };
}

function versionConstraint() {
  return {
    mode: 'all' as const,
    version_numbers: [],
    active: false,
    confidence: 0,
    rationale: 'No version constraint.',
    detector_name: 'none',
  };
}

function preference(): DocumentPreference {
  return {
    document: {
      key: 'document:deployment-guide',
      display_name: 'deployment-guide.pdf',
      document_id: 'document-1',
      document_version_ids: ['version-1'],
      normalized_names: ['deployment guide'],
    },
    score: 0.9,
    confidence: 0.86,
    margin: 0.2,
    supporting_information_need_ids: ['need-1'],
    supporting_evidence_ranks: [1],
    rationale: 'The deployment guide is the strongest source.',
    detector_name: 'test',
    semantics: 'soft_preference_not_filter',
  };
}
