import { AgentTraceComponent } from './agent-trace.component';
import { QueryResponse } from '../../models/query.models';

describe('AgentTraceComponent', () => {
  it('summarizes per-information-need plans when no top-level plan is recorded', () => {
    const component = new AgentTraceComponent();
    component.result = buildQueryResponse();

    expect(component.primaryStrategyLabel()).toBe('multi query');
    expect(component.primaryPipelineName()).toBe('agentic-rag');
    expect(component.totalAttempts()).toBe(2);
    expect(component.retryCount()).toBe(1);
    expect(component.resolvedNeedsLabel()).toBe('1/1');
    expect(component.percentage(component.overallCoverage() ?? 0)).toBe(82);
  });
});

function buildQueryResponse(): QueryResponse {
  const classification = {
    query_type: 'broad_explanation' as const,
    confidence: 0.9,
    needs_metadata_filters: false,
    metadata_filter_hints: [],
    rationale: 'The question asks for a broad explanation.',
    classifier_name: 'heuristic',
    fallback_used: false,
    document_constraint: {
      names: [],
      normalized_names: [],
      active: false,
      confidence: 0,
      rationale: 'No document constraint.',
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
  };

  const plan = {
    information_need_id: 'need-1',
    strategy: 'multi_query' as const,
    selected_pipeline_name: 'agentic-rag',
    query: 'Explain the project purpose',
    top_k: 8,
    rationale: 'Use query variants for broad coverage.',
    planner_name: 'rules',
    based_on_query_type: 'broad_explanation' as const,
    attempt_number: 2,
    metadata_filter_hints: [],
    requires_reranking: false,
    adjustments: ['broaden_query'],
    document_constraint: classification.document_constraint,
    version_constraint: classification.version_constraint,
    date_constraints: [],
    preferred_document: null,
  };

  const finalGrade = {
    information_need_id: 'need-1',
    description: 'Explain the project purpose',
    status: 'supported' as const,
    coverage_score: 0.82,
    supporting_evidence_ranks: [1],
    rationale: 'The retrieved passage directly explains the purpose.',
    required: true,
  };

  return {
    id: 'run-1',
    question: 'What is the project for?',
    answer: 'It indexes and retrieves document knowledge.',
    status: 'succeeded',
    pipeline_name: 'agentic-rag',
    pipeline_version: '1.0.0',
    top_k: 5,
    started_at: null,
    completed_at: null,
    error_message: null,
    classification,
    information_need_decomposition: {
      information_needs: [
        {
          need_id: 'need-1',
          description: 'Explain the project purpose',
          retrieval_query: 'Explain the project purpose',
          required: true,
        },
      ],
      information_need_count: 1,
      rationale: 'One atomic need is sufficient.',
      decomposer_name: 'heuristic',
      fallback_used: false,
    },
    retrieval_plan: null,
    evidence_grading: null,
    retrieval_retry: null,
    primary_document_preference: null,
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
            description: 'Explain the project purpose',
            retrieval_query: 'Explain the project purpose',
            required: true,
          },
          information_need_id: 'need-1',
          status: 'supported',
          attempts_used: 2,
          max_attempts: 3,
          reclassifications_used: 0,
          parent_information_need_id: null,
          depth: 0,
          classification,
          classification_history: [classification],
          current_plan: plan,
          plan_history: [plan],
          attempts: [],
          constraint_validation_history: [],
          evidence_keys: ['chunk-1'],
          final_grade: finalGrade,
          stop_reason: 'supported',
          stop_rationale: 'Evidence is sufficient.',
        },
      ],
    },
    constraint_validation: null,
    evidence_context: null,
    evidence: [],
    citations: [],
    trace: [],
  };
}
