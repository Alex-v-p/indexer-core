import { TestBed } from '@angular/core/testing';

import {
  DocumentPreference,
  EvidenceGrading,
  InformationNeed,
  InformationNeedAttempt,
  InformationNeedDecomposition,
  InformationNeedResolution,
  TraceStep,
} from '../../models/query.models';
import { DocumentPreferenceTraceComponent } from './document-preference-trace.component';

describe('DocumentPreferenceTraceComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DocumentPreferenceTraceComponent],
    }).compileComponents();
  });

  it('links preference events to need descriptions and identifies inherited preferences', () => {
    const component = new DocumentPreferenceTraceComponent();
    component.preference = preference();
    component.decomposition = decomposition();
    component.resolution = resolution();
    component.aggregateEvidence = aggregateEvidence();
    component.trace = preferenceTrace();

    const events = component.detectionEvents();

    expect(events).toHaveLength(2);
    expect(events[0].need.description).toBe('Identify deployment requirements');
    expect(events[0].need.retrieval_query).toBe('production deployment requirements');
    expect(events[0].selectedDocument).toBe('deployment-guide.pdf');
    expect(events[0].supportingEvidence[0].text).toContain('rolling deployment');
    expect(events[1].need.description).toBe('Explain fallback behavior');
    expect(events[1].inherited).toBe(true);
    expect(events[1].previousDocument).toBe('deployment-guide.pdf');
  });

  it('renders retrieval, balancing, evidence grading, and distinct retention states', () => {
    const fixture = TestBed.createComponent(DocumentPreferenceTraceComponent);
    fixture.componentInstance.preference = preference();
    fixture.componentInstance.decomposition = decomposition();
    fixture.componentInstance.resolution = resolution();
    fixture.componentInstance.aggregateEvidence = aggregateEvidence();
    fixture.componentInstance.trace = preferenceTrace();
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    const attempts = element.querySelectorAll('[data-testid="balanced-attempt"]');
    const text = element.textContent ?? '';

    expect(attempts).toHaveLength(1);
    expect(text).toContain('Identify deployment requirements');
    expect(text).toContain('need-1 · attempt 1');
    expect(text).toContain('production deployment requirements');
    expect(text).toContain('hybrid-rag');
    expect(text).toContain('reciprocal rank fusion');
    expect(text).toContain('Candidate top-k · 8');
    expect(text).toContain('8');
    expect(text).toContain('2');
    expect(text).toContain('Document quota relaxed');
    expect(text).toContain('Deployment uses a rolling deployment with health checks.');
    expect(text).toContain('This unrelated billing paragraph does not explain deployment.');
    expect(text).toContain('Accepted for this need');
    expect(text).toContain('Rejected for this need');
    expect(text).toContain('Retained after need grading');
    expect(text).toContain('Removed after need grading');
    expect(text).toContain('Retained in aggregated evidence');
    expect(text).toContain('Not retained in aggregated evidence');
  });

  it('renders an older trace safely when structured attempts are absent', () => {
    const fixture = TestBed.createComponent(DocumentPreferenceTraceComponent);
    fixture.componentInstance.decomposition = decomposition();
    fixture.componentInstance.trace = [
      {
        id: null,
        step_order: 12,
        name: 'execute_information_need_plan',
        step_type: 'information_need_retrieval',
        status: 'completed',
        duration_ms: 12,
        input_summary: null,
        output_summary: null,
        error_message: null,
        metadata: {
          active_information_need_lookup: {
            information_need_id: 'need-1',
            attempt_number: 1,
            document_balancing: {
              selector_name: 'legacy_selector',
              candidate_count: 5,
              requested_top_k: 2,
              selected_count: 2,
              selected_document_counts: {},
              primary_document_key: null,
              primary_selected_count: 0,
              quota_relaxed: false,
            },
          },
        },
      },
    ];

    expect(() => fixture.detectChanges()).not.toThrow();
    expect(fixture.nativeElement.textContent).toContain(
      'Evidence snapshots were not stored for this older run.',
    );
  });
});

function decomposition(): InformationNeedDecomposition {
  return {
    information_needs: [primaryNeed(), fallbackNeed()],
    information_need_count: 2,
    rationale: 'Separate deployment behavior from fallback behavior.',
    decomposer_name: 'test',
    fallback_used: false,
  };
}

function primaryNeed(): InformationNeed {
  return {
    need_id: 'need-1',
    description: 'Identify deployment requirements',
    retrieval_query: 'production deployment requirements',
    required: true,
  };
}

function fallbackNeed(): InformationNeed {
  return {
    need_id: 'need-2',
    description: 'Explain fallback behavior',
    retrieval_query: 'deployment fallback behavior',
    required: false,
  };
}

function resolution(): InformationNeedResolution {
  const currentAttempt = attempt();
  return {
    graph_name: 'information_need_resolution',
    information_need_count: 2,
    supported_information_need_ids: ['need-1'],
    supported_information_need_count: 1,
    unresolved_information_need_ids: ['need-2'],
    unresolved_information_need_count: 1,
    complete: false,
    total_retrieval_attempts: 1,
    max_total_retrieval_attempts: 8,
    max_attempts_per_information_need: 3,
    executions: [
      {
        information_need: primaryNeed(),
        information_need_id: 'need-1',
        status: 'supported',
        attempts_used: 1,
        max_attempts: 3,
        reclassifications_used: 0,
        parent_information_need_id: null,
        depth: 0,
        classification: null,
        classification_history: [],
        current_plan: currentAttempt.plan,
        plan_history: [currentAttempt.plan],
        attempts: [currentAttempt],
        constraint_validation_history: [],
        evidence_keys: ['qdrant_chunk:accepted'],
        final_grade: currentAttempt.evidence_grading.information_need_grades[0],
        stop_reason: 'supported',
        stop_rationale: 'The need is covered.',
      },
      {
        information_need: fallbackNeed(),
        information_need_id: 'need-2',
        status: 'exhausted',
        attempts_used: 0,
        max_attempts: 3,
        reclassifications_used: 0,
        parent_information_need_id: null,
        depth: 0,
        classification: null,
        classification_history: [],
        current_plan: null,
        plan_history: [],
        attempts: [],
        constraint_validation_history: [],
        evidence_keys: [],
        final_grade: null,
        stop_reason: 'exhausted',
        stop_rationale: 'No more attempts.',
      },
    ],
  };
}

function attempt(): InformationNeedAttempt {
  const plan = {
    information_need_id: 'need-1',
    strategy: 'hybrid' as const,
    selected_pipeline_name: 'hybrid-rag',
    query: 'production deployment requirements',
    top_k: 2,
    rationale: 'Fuse semantic and keyword evidence, then balance document coverage.',
    planner_name: 'test',
    based_on_query_type: 'broad_explanation' as const,
    attempt_number: 1,
    metadata_filter_hints: [],
    requires_reranking: true,
    adjustments: ['prefer_primary_document'],
    document_constraint: {
      names: [],
      normalized_names: [],
      active: false,
      confidence: 0,
      rationale: 'No constraint.',
      detector_name: 'none',
      match_semantics: 'exact_normalized_any' as const,
    },
    version_constraint: {
      mode: 'all' as const,
      version_numbers: [],
      active: false,
      confidence: 0,
      rationale: 'No version constraint.',
      detector_name: 'none',
    },
    date_constraints: [],
    preferred_document: preference(),
  };
  const grading = needGrading();
  return {
    attempt_number: 1,
    query: plan.query,
    top_k: 2,
    pipeline_name: plan.selected_pipeline_name,
    strategy: plan.strategy,
    adjustments: plan.adjustments,
    retrieved_count: 2,
    unique_evidence_added: 2,
    evidence_keys: ['qdrant_chunk:accepted'],
    evidence: [
      {
        evidence_key: 'qdrant_chunk:accepted',
        retrieval_order: 1,
        aggregate_rank: 1,
        text: 'Deployment uses a rolling deployment with health checks.',
        score: 0.93,
        qdrant_chunk_index_id: 'accepted',
        document_id: 'document-1',
        document_version_id: 'version-2',
        metadata: {
          original_filename: 'deployment-guide.pdf',
          document_version_label: 'v2',
          page_number: 7,
          section_title: 'Rollout',
          chunk_ordinal: 12,
          document_balancing: { primary_document: true },
        },
        relevance_score: 0.91,
        relevant: true,
        grading_rationale: 'Directly describes the requested deployment behavior.',
        supports_information_need_ids: ['need-1'],
        retained_after_need_grading: true,
      },
      {
        evidence_key: 'qdrant_chunk:rejected',
        retrieval_order: 2,
        aggregate_rank: 2,
        text: 'This unrelated billing paragraph does not explain deployment.',
        score: 0.51,
        qdrant_chunk_index_id: 'rejected',
        document_id: 'document-2',
        document_version_id: 'version-1',
        metadata: {
          original_filename: 'billing-guide.pdf',
          page_number: 2,
          document_balancing: { primary_document: false },
        },
        relevance_score: 0.12,
        relevant: false,
        grading_rationale: 'The paragraph is unrelated to deployment.',
        supports_information_need_ids: [],
        retained_after_need_grading: false,
      },
    ],
    retrieval_metadata: {
      requested_top_k: 2,
      candidate_top_k: 8,
      retriever_type: 'hybrid',
      fusion_method: 'reciprocal_rank',
      vector_contribution: 0.6,
      keyword_contribution: 0.4,
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
      applied: true,
      provider: 'cross-encoder',
      candidate_count_before: 8,
      candidate_count_after: 2,
      details: {},
    },
    document_balancing: {
      selector_name: 'document_balanced_candidate_selector',
      candidate_count: 8,
      requested_top_k: 2,
      selected_count: 2,
      selected_chunks_per_document: {
        'document:deployment-guide': 1,
        'document:billing-guide': 1,
      },
      primary_document_key: 'document:deployment-guide',
      primary_document_quota: 2,
      primary_document_selected_count: 1,
      quota_relaxed: true,
      preferred_document_active: true,
      details: {},
    },
    plan,
    constraint_validation: {
      status: 'not_requested',
      blocked: false,
      candidate_count: 2,
      matched_count: 2,
      rejected_count: 0,
      constraints: {},
      rationale: 'No constraints.',
    },
    evidence_grading: grading,
  };
}

function needGrading(): EvidenceGrading {
  return {
    status: 'sufficient',
    coverage_score: 0.9,
    sufficient: true,
    answerable: true,
    partial_answer_available: false,
    missing_evidence: false,
    weak_evidence: false,
    relevant_count: 1,
    total_count: 2,
    relevant_evidence_ranks: [1],
    supported_information_need_count: 1,
    supported_required_information_need_count: 1,
    required_information_need_count: 1,
    supported_information: ['Deployment requirements'],
    partial_information_need_count: 0,
    missing_information_need_count: 0,
    total_information_need_count: 1,
    unresolved_information: [],
    rationale: 'One evidence item directly supports the need.',
    grader_name: 'test',
    fallback_used: false,
    grades: [
      {
        evidence_rank: 1,
        relevance_score: 0.91,
        relevant: true,
        rationale: 'Direct support.',
        supports_information_need_ids: ['need-1'],
      },
      {
        evidence_rank: 2,
        relevance_score: 0.12,
        relevant: false,
        rationale: 'Unrelated.',
        supports_information_need_ids: [],
      },
    ],
    information_need_grades: [
      {
        information_need_id: 'need-1',
        description: 'Identify deployment requirements',
        status: 'supported',
        coverage_score: 0.9,
        supporting_evidence_ranks: [1],
        rationale: 'The deployment evidence resolves the need.',
        required: true,
      },
    ],
  };
}

function aggregateEvidence() {
  return [
    {
      id: 'evidence-1',
      rank: 1,
      score: 0.93,
      text: 'Deployment uses a rolling deployment with health checks.',
      qdrant_chunk_index_id: 'accepted',
      document_id: 'document-1',
      document_version_id: 'version-2',
      metadata: { original_filename: 'deployment-guide.pdf' },
    },
  ];
}

function preference(): DocumentPreference {
  return {
    document: {
      key: 'document:deployment-guide',
      display_name: 'deployment-guide.pdf',
      document_id: 'document-1',
      document_version_ids: ['version-2'],
      normalized_names: ['deployment guide'],
    },
    score: 0.92,
    confidence: 0.88,
    margin: 0.2,
    supporting_information_need_ids: ['need-1'],
    supporting_evidence_ranks: [1],
    rationale: 'The deployment guide has the strongest directly graded support.',
    detector_name: 'test-detector',
    semantics: 'soft_preference_not_filter',
  };
}

function preferenceTrace(): TraceStep[] {
  return [
    preferenceStep(8, 'need-1', null, true),
    preferenceStep(18, 'need-2', 'document:deployment-guide', false),
  ];
}

function preferenceStep(
  stepOrder: number,
  needId: string,
  previousDocumentKey: string | null,
  changed: boolean,
): TraceStep {
  return {
    id: null,
    step_order: stepOrder,
    name: 'detect_primary_document',
    step_type: 'document_preference',
    status: 'completed',
    duration_ms: 2,
    input_summary: null,
    output_summary: null,
    error_message: null,
    metadata: {
      primary_document_detection: {
        information_need_id: needId,
        detector_name: 'test-detector',
        candidate_evidence_count: 2,
        previous_document_key: previousDocumentKey,
        selected_document_key: 'document:deployment-guide',
        preference_changed: changed,
      },
      primary_document_preference: preference(),
    },
  };
}
