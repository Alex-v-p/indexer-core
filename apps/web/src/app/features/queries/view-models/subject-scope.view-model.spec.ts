import { buildIntegratedTraceResponse } from '../testing/agent-trace-test.fixture';
import { buildSubjectScopeViewModel } from './subject-scope.view-model';

describe('buildSubjectScopeViewModel', () => {
  it('keeps historical runs without scope readable', () => {
    const result = buildIntegratedTraceResponse();
    result.subject_scope = null;
    expect(buildSubjectScopeViewModel(result)).toBeNull();
  });

  it('summarizes comparison lanes and effective coverage', () => {
    const result = buildIntegratedTraceResponse();
    result.subject_scope = {
      requested_subject_ids: ['alpha', 'beta'],
      matched_subject_ids: ['alpha', 'beta'],
      catalog: [
        { subject_id: 'alpha', kind: 'project', name: 'Alpha', aliases: [] },
        { subject_id: 'beta', kind: 'project', name: 'Beta', aliases: [] },
      ],
      document_scope: { strict: true, global: false, strict_empty: false, allowed_document_ids: ['a', 'b'], allowed_document_count: 2 },
      source: 'explicit', confidence: 1, strict: true, catalog_revision: '1', policy_revision: '1',
      coverage_mode: 'multi_document', product_outcome: 'answered', clarification_reason: null,
      ambiguity_candidates: [], comparison_requested: true,
      subject_lanes: [
        { lane_id: 'subject:alpha', subject_id: 'alpha', subject_name: 'Alpha', document_scope: { strict: true, global: false, strict_empty: false, allowed_document_ids: ['a'], allowed_document_count: 1 } },
        { lane_id: 'subject:beta', subject_id: 'beta', subject_name: 'Beta', document_scope: { strict: true, global: false, strict_empty: false, allowed_document_ids: ['b'], allowed_document_count: 1 } },
      ],
    };
    const resolution = result.information_need_resolution;
    const grading = result.evidence_grading;
    if (!resolution || !grading) {
      throw new Error('The integrated fixture must include retrieval resolution and grading.');
    }
    const execution = resolution.executions[0];
    const plan = execution?.current_plan;
    if (!execution || !plan) {
      throw new Error('The integrated fixture must include an active information-need plan.');
    }
    execution.information_need.subject_lane = { lane_id: 'subject:alpha', subject_id: 'alpha', subject_name: 'Alpha' };
    execution.attempts = [{
      attempt_number: 1,
      query: 'Alpha delivery',
      top_k: 5,
      pipeline_name: 'agentic_rag',
      strategy: 'hybrid',
      adjustments: [],
      retrieved_count: 1,
      unique_evidence_added: 1,
      evidence_keys: ['evidence-1'],
      evidence: [],
      retrieval_metadata: {
        requested_top_k: 5, candidate_top_k: 5, retriever_type: 'hybrid', fusion_method: 'rrf',
        vector_contribution: 1, keyword_contribution: 1, query_variants: [], multi_query_query_count: null,
        multi_query_candidate_top_k_per_query: null, multi_query_result_counts: {},
        hierarchical_document_candidate_count: null, hierarchical_selected_document_version_ids: [],
        hierarchical_section_candidate_count: null, hierarchical_selected_section_ids: [], details: {},
        coverage: {
          requested_mode: 'multi_document', effective_mode: 'best_evidence', fallback_reason: 'single_document_scope',
          searched_document_ids: ['a'], contributing_document_ids: ['a'], searched_document_count: 1,
          contributing_document_count: 1, coverage_satisfied: true, out_of_scope_rejected_count: 2,
        },
      },
      reranking_metadata: { applied: false, provider: null, candidate_count_before: null, candidate_count_after: null, details: {} },
      document_balancing: {
        selector_name: null, candidate_count: null, requested_top_k: null, selected_count: null,
        selected_chunks_per_document: {}, primary_document_key: null, primary_document_quota: null,
        primary_document_selected_count: null, quota_relaxed: false, preferred_document_active: false, details: {},
      },
      plan,
      constraint_validation: { status: 'matched', blocked: false, candidate_count: 1, matched_count: 1, rejected_count: 0, constraints: {}, rationale: 'Lane scope matched.' },
      evidence_grading: grading,
    }];

    const view = buildSubjectScopeViewModel(result)!;
    expect(view.effectiveCoverage).toBe('best_evidence');
    expect(view.fallback).toBe('single_document_scope');
    expect(view.lanes[0]).toMatchObject({ subjectName: 'Alpha', searchedDocumentCount: 1, contributingDocumentCount: 1, coverageSatisfied: true, rejectedCount: 2 });
    expect(view.lanes[1]).toMatchObject({ subjectName: 'Beta', support: 'missing' });
  });
});
