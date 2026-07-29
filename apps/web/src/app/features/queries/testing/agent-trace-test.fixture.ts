import {
  DocumentPreference,
  EvidenceGrading,
  InformationNeedRetrievalPlan,
  QueryResponse,
} from '../models/query.models';

export function buildIntegratedTraceResponse(): QueryResponse {
  const firstPlan = retrievalPlan(1, 'hybrid', 'hybrid-rag', []);
  const secondPlan = retrievalPlan(
    2,
    'multi_query',
    'multi-query-rag',
    ['broaden_query', 'increase_top_k'],
  );
  secondPlan.preferred_document = preference();
  const finalGrade = {
    information_need_id: 'need-1',
    description: 'Explain deployment behavior',
    status: 'supported' as const,
    coverage_score: 0.9,
    supporting_evidence_ranks: [],
    rationale: 'The broadened lookup resolved the deployment need.',
    required: true,
  };

  return {
    id: 'integrated-run',
    question: 'How does deployment work?',
    answer: 'Deployment uses a controlled rollout.',
    status: 'succeeded',
    pipeline_name: 'agentic-rag',
    pipeline_version: '1.0.0',
    top_k: 5,
    started_at: null,
    completed_at: null,
    error_message: null,
    classification: {
      query_type: 'broad_explanation',
      confidence: 0.91,
      needs_metadata_filters: false,
      metadata_filter_hints: [],
      rationale: 'The question requests a broad operational explanation.',
      classifier_name: 'test-classifier',
      fallback_used: false,
      document_constraint: documentConstraint(),
      version_constraint: versionConstraint(),
      date_constraints: [],
    },
    information_need_decomposition: {
      information_needs: [
        {
          need_id: 'need-1',
          description: 'Explain deployment behavior',
          retrieval_query: 'deployment behavior',
          required: true,
        },
      ],
      information_need_count: 1,
      rationale: 'A single atomic need covers the question.',
      decomposer_name: 'test-decomposer',
      fallback_used: false,
    },
    retrieval_plan: {
      strategy: 'baseline',
      selected_pipeline_name: 'baseline-rag',
      rationale: 'Begin with a query-level baseline before need-specific planning.',
      planner_name: 'test-planner',
      based_on_query_type: 'broad_explanation',
      metadata_filter_hints: [],
      requires_reranking: false,
      target_information_need_ids: ['need-1'],
      target_information_need_count: 1,
      document_constraint: documentConstraint(),
      version_constraint: versionConstraint(),
      date_constraints: [],
      preferred_document: null,
    },
    evidence_grading: finalGrading(),
    retrieval_retry: {
      policy_name: 'bounded_retry',
      max_retries: 2,
      retries_used: 1,
      attempt_count: 2,
      claim_plan_count: 0,
      claim_lookup_count: 0,
      stop_reason: 'sufficient',
      stop_rationale: 'The retry resolved the need.',
      final_sufficient: true,
      final_pipeline_name: 'multi-query-rag',
      final_strategy: 'multi_query',
      final_top_k: 10,
      final_query: 'production deployment behavior',
      query_changed: true,
      attempts: [],
    },
    primary_document_preference: preference(),
    information_need_resolution: {
      graph_name: 'information_need_resolution',
      information_need_count: 1,
      supported_information_need_ids: ['need-1'],
      supported_information_need_count: 1,
      unresolved_information_need_ids: [],
      unresolved_information_need_count: 0,
      complete: true,
      total_retrieval_attempts: 2,
      max_total_retrieval_attempts: 8,
      max_attempts_per_information_need: 3,
      executions: [
        {
          information_need: {
            need_id: 'need-1',
            description: 'Explain deployment behavior',
            retrieval_query: 'deployment behavior',
            required: true,
          },
          information_need_id: 'need-1',
          status: 'supported',
          attempts_used: 2,
          max_attempts: 3,
          reclassifications_used: 0,
          parent_information_need_id: null,
          depth: 0,
          classification: null,
          classification_history: [],
          current_plan: secondPlan,
          plan_history: [secondPlan, firstPlan],
          attempts: [],
          constraint_validation_history: [],
          evidence_keys: [],
          final_grade: finalGrade,
          stop_reason: 'supported',
          stop_rationale: 'Evidence coverage reached the required threshold.',
        },
      ],
    },
    constraint_validation: null,
    evidence_context: null,
    evidence: [],
    citations: [],
    trace: [
      {
        id: 'trace-1',
        step_order: 1,
        name: 'classify_query',
        step_type: 'classification',
        status: 'completed',
        duration_ms: 2,
        input_summary: 'How does deployment work?',
        output_summary: 'broad explanation',
        error_message: null,
        metadata: {},
      },
      {
        id: 'trace-2',
        step_order: 20,
        name: 'arbitrate_final_evidence',
        step_type: 'evidence_arbitration',
        status: 'completed',
        duration_ms: 3,
        input_summary: null,
        output_summary: 'sufficient evidence',
        error_message: null,
        metadata: { evidence_arbitration: { status: 'sufficient' } },
      },
    ],
  };
}

export function buildPartialTraceResponse(): QueryResponse {
  const result = buildIntegratedTraceResponse();
  return {
    ...result,
    classification: null,
    information_need_decomposition: null,
    retrieval_plan: null,
    evidence_grading: null,
    retrieval_retry: null,
    primary_document_preference: null,
    information_need_resolution: null,
    evidence: [],
    citations: [],
    trace: [],
  };
}

function retrievalPlan(
  attemptNumber: number,
  strategy: InformationNeedRetrievalPlan['strategy'],
  pipeline: string,
  adjustments: string[],
): InformationNeedRetrievalPlan {
  return {
    information_need_id: 'need-1',
    strategy,
    selected_pipeline_name: pipeline,
    query:
      attemptNumber === 1
        ? 'deployment behavior'
        : 'production deployment behavior',
    top_k: attemptNumber === 1 ? 5 : 10,
    rationale:
      attemptNumber === 1
        ? 'Start with hybrid retrieval.'
        : 'Broaden the query after weak initial coverage.',
    planner_name: 'test-planner',
    based_on_query_type: 'broad_explanation',
    attempt_number: attemptNumber,
    metadata_filter_hints: [],
    requires_reranking: false,
    adjustments,
    document_constraint: documentConstraint(),
    version_constraint: versionConstraint(),
    date_constraints: [],
    preferred_document: null,
  };
}

function finalGrading(): EvidenceGrading {
  return {
    status: 'sufficient',
    coverage_score: 0.9,
    sufficient: true,
    answerable: true,
    partial_answer_available: false,
    missing_evidence: false,
    weak_evidence: false,
    relevant_count: 0,
    total_count: 0,
    relevant_evidence_ranks: [],
    supported_information_need_count: 1,
    supported_required_information_need_count: 1,
    required_information_need_count: 1,
    supported_information: ['Deployment behavior'],
    partial_information_need_count: 0,
    missing_information_need_count: 0,
    total_information_need_count: 1,
    unresolved_information: [],
    rationale: 'The final evidence covers the deployment need.',
    grader_name: 'test-arbitrator',
    fallback_used: false,
    grades: [],
    information_need_grades: [
      {
        information_need_id: 'need-1',
        description: 'Explain deployment behavior',
        status: 'supported',
        coverage_score: 0.9,
        supporting_evidence_ranks: [],
        rationale: 'The deployment need is resolved.',
        required: true,
      },
    ],
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
    supporting_evidence_ranks: [],
    rationale: 'The deployment guide is the strongest source.',
    detector_name: 'test-detector',
    semantics: 'soft_preference_not_filter',
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
