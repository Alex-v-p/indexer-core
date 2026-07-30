export interface VersionConstraint {
  mode:
    | 'all'
    | 'all_versions'
    | 'latest'
    | 'oldest'
    | 'specific'
    | 'previous'
    | 'all_except_latest'
    | 'latest_and_previous'
    | 'oldest_and_latest';
  version_numbers: number[];
  active: boolean;
  confidence: number;
  rationale: string;
  detector_name: string;
}
export interface DocumentNameConstraint {
  names: string[];
  normalized_names: string[];
  active: boolean;
  confidence: number;
  rationale: string;
  detector_name: string;
  match_semantics: 'exact_normalized_any';
}
export interface DocumentReference {
  key: string;
  display_name: string;
  document_id: string | null;
  document_version_ids: string[];
  normalized_names: string[];
}
export interface DocumentPreference {
  document: DocumentReference;
  score: number;
  confidence: number;
  margin: number;
  supporting_information_need_ids: string[];
  supporting_evidence_ranks: number[];
  rationale: string;
  detector_name: string;
  semantics: 'soft_preference_not_filter' | string;
}
export interface DocumentBalancing {
  selector_name: string;
  candidate_count: number;
  requested_top_k: number;
  selected_count: number;
  selected_document_counts: Record<string, number>;
  primary_document_key: string | null;
  primary_selected_count: number;
  quota_relaxed: boolean;
}
export interface PrimaryDocumentDetection {
  information_need_id: string;
  detector_name: string;
  candidate_evidence_count: number;
  previous_document_key: string | null;
  selected_document_key: string | null;
  preference_changed: boolean;
}
export interface DateRangeConstraint {
  start: string | null;
  end: string | null;
  end_exclusive: boolean;
}
export interface DateConstraint {
  field: 'uploaded_at' | 'published_at' | 'any_recorded_at';
  range: DateRangeConstraint;
  original_expression: string;
  active: boolean;
  confidence: number;
  rationale: string;
  detector_name: string;
}
export interface ConstraintValidation {
  status: 'not_requested' | 'matched' | 'no_match';
  blocked: boolean;
  candidate_count: number;
  matched_count: number;
  rejected_count: number;
  constraints: Record<string, unknown>;
  rationale: string;
}
export interface EvidenceSourceContext {
  evidence_rank: number;
  values: Record<string, string>;
}
export interface EvidenceContext {
  constraint_summary: string;
  constraints: Record<string, unknown>;
  validation: ConstraintValidation;
  sources: EvidenceSourceContext[];
}
export interface QueryClassification {
  query_type: 'factual_lookup' | 'broad_explanation' | 'comparison' | 'version_specific';
  confidence: number;
  needs_metadata_filters: boolean;
  metadata_filter_hints: string[];
  rationale: string;
  classifier_name: string;
  fallback_used: boolean;
  document_constraint: DocumentNameConstraint;
  version_constraint: VersionConstraint;
  date_constraints: DateConstraint[];
}
export interface InformationNeed {
  need_id: string;
  description: string;
  retrieval_query: string;
  required: boolean;
}
export interface InformationNeedDecomposition {
  information_needs: InformationNeed[];
  information_need_count: number;
  rationale: string;
  decomposer_name: string;
  fallback_used: boolean;
}
export interface RetrievalPlan {
  strategy: 'baseline' | 'hybrid' | 'contextual' | 'hierarchical' | 'multi_query' | 'rerank';
  selected_pipeline_name: string;
  rationale: string;
  planner_name: string;
  based_on_query_type: QueryClassification['query_type'];
  metadata_filter_hints: string[];
  requires_reranking: boolean;
  target_information_need_ids: string[];
  target_information_need_count: number;
  document_constraint: DocumentNameConstraint;
  version_constraint: VersionConstraint;
  date_constraints: DateConstraint[];
  preferred_document: DocumentPreference | null;
}
export interface InformationNeedGrade {
  information_need_id: string;
  description: string;
  status: 'missing' | 'partial' | 'supported';
  coverage_score: number;
  supporting_evidence_ranks: number[];
  rationale: string;
  required: boolean;
}
export interface EvidenceGrade {
  evidence_rank: number;
  relevance_score: number;
  relevant: boolean;
  rationale: string;
  supports_information_need_ids: string[];
}
export interface EvidenceGrading {
  status: 'missing' | 'weak' | 'sufficient';
  coverage_score: number;
  sufficient: boolean;
  answerable: boolean;
  partial_answer_available: boolean;
  missing_evidence: boolean;
  weak_evidence: boolean;
  relevant_count: number;
  total_count: number;
  relevant_evidence_ranks: number[];
  supported_information_need_count: number;
  supported_required_information_need_count: number;
  required_information_need_count: number;
  supported_information: string[];
  partial_information_need_count: number;
  missing_information_need_count: number;
  total_information_need_count: number;
  unresolved_information: string[];
  rationale: string;
  grader_name: string;
  fallback_used: boolean;
  grades: EvidenceGrade[];
  information_need_grades: InformationNeedGrade[];
}
export interface ClaimRetrievalTask {
  information_need_id: string;
  description: string;
  retrieval_query: string;
  prior_status: string;
  prior_coverage_score: number;
  prior_supporting_evidence_ranks: number[];
  grading_feedback: string;
  rationale: string;
}
export interface ClaimRetrievalPlan {
  planner_name: string;
  rationale: string;
  target_information_need_ids: string[];
  target_information_need_count: number;
  deferred_information_need_ids: string[];
  deferred_information_need_count: number;
  tasks: ClaimRetrievalTask[];
}
export interface ClaimLookup {
  information_need_id: string;
  query: string;
  pipeline_name: string;
  strategy: RetrievalPlan['strategy'];
  top_k: number;
  retrieved_count: number;
  unique_evidence_added: number;
}
export interface RetrievalAttempt {
  attempt_number: number;
  retry_number: number;
  query: string;
  top_k: number;
  pipeline_name: string;
  strategy: RetrievalPlan['strategy'];
  evidence_count: number;
  new_evidence_count: number;
  accumulated_evidence_count: number;
  actions: string[];
  decision_rationale: string | null;
  resolved_information_need_ids: string[];
  remaining_information_need_ids: string[];
  claim_retrieval_plan: ClaimRetrievalPlan | null;
  claim_lookups: ClaimLookup[];
  retrieval_plan: RetrievalPlan;
  evidence_grading: EvidenceGrading;
}
export interface RetrievalRetry {
  policy_name: string;
  max_retries: number;
  retries_used: number;
  attempt_count: number;
  claim_plan_count: number;
  claim_lookup_count: number;
  stop_reason: string;
  stop_rationale: string;
  final_sufficient: boolean;
  final_pipeline_name: string;
  final_strategy: RetrievalPlan['strategy'];
  final_top_k: number;
  final_query: string;
  query_changed: boolean;
  attempts: RetrievalAttempt[];
}
export interface InformationNeedRetrievalPlan {
  information_need_id: string;
  strategy: RetrievalPlan['strategy'];
  selected_pipeline_name: string;
  query: string;
  top_k: number;
  rationale: string;
  planner_name: string;
  based_on_query_type: QueryClassification['query_type'];
  attempt_number: number;
  metadata_filter_hints: string[];
  requires_reranking: boolean;
  adjustments: string[];
  document_constraint: DocumentNameConstraint;
  version_constraint: VersionConstraint;
  date_constraints: DateConstraint[];
  preferred_document: DocumentPreference | null;
}
export interface InformationNeedAttemptEvidence {
  evidence_key: string;
  retrieval_order: number;
  aggregate_rank: number | null;
  text: string;
  score: number | null;
  qdrant_chunk_index_id: string | null;
  document_id: string | null;
  document_version_id: string | null;
  metadata: Record<string, unknown>;
  relevance_score: number | null;
  relevant: boolean | null;
  grading_rationale: string | null;
  supports_information_need_ids: string[];
  retained_after_need_grading: boolean;
}
export interface RetrievalExecutionMetadata {
  requested_top_k: number | null;
  candidate_top_k: number | null;
  retriever_type: string | null;
  fusion_method: string | null;
  vector_contribution: number | null;
  keyword_contribution: number | null;
  query_variants: string[];
  multi_query_query_count: number | null;
  multi_query_candidate_top_k_per_query: number | null;
  multi_query_result_counts: Record<string, number>;
  hierarchical_document_candidate_count: number | null;
  hierarchical_selected_document_version_ids: string[];
  hierarchical_section_candidate_count: number | null;
  hierarchical_selected_section_ids: string[];
  details: Record<string, unknown>;
}
export interface RerankingMetadata {
  applied: boolean;
  provider: string | null;
  candidate_count_before: number | null;
  candidate_count_after: number | null;
  details: Record<string, unknown>;
}
export interface AttemptDocumentBalancing {
  selector_name: string | null;
  candidate_count: number | null;
  requested_top_k: number | null;
  selected_count: number | null;
  selected_chunks_per_document: Record<string, number>;
  primary_document_key: string | null;
  primary_document_quota: number | null;
  primary_document_selected_count: number | null;
  quota_relaxed: boolean;
  preferred_document_active: boolean;
  details: Record<string, unknown>;
}
export interface InformationNeedAttempt {
  attempt_number: number;
  query: string;
  top_k: number;
  pipeline_name: string;
  strategy: RetrievalPlan['strategy'];
  adjustments: string[];
  retrieved_count: number;
  unique_evidence_added: number;
  evidence_keys: string[];
  evidence: InformationNeedAttemptEvidence[];
  retrieval_metadata: RetrievalExecutionMetadata;
  reranking_metadata: RerankingMetadata;
  document_balancing: AttemptDocumentBalancing;
  plan: InformationNeedRetrievalPlan;
  constraint_validation: ConstraintValidation;
  evidence_grading: EvidenceGrading;
}
export interface InformationNeedExecution {
  information_need: InformationNeed;
  information_need_id: string;
  status: 'pending' | 'active' | 'supported' | 'exhausted' | 'failed';
  attempts_used: number;
  max_attempts: number;
  reclassifications_used: number;
  parent_information_need_id: string | null;
  depth: number;
  classification: QueryClassification | null;
  classification_history: QueryClassification[];
  classification_source_history?: ('top_level_reuse' | 'model')[];
  current_plan: InformationNeedRetrievalPlan | null;
  plan_history: InformationNeedRetrievalPlan[];
  attempts: InformationNeedAttempt[];
  constraint_validation_history: ConstraintValidation[];
  evidence_keys: string[];
  final_grade: InformationNeedGrade | null;
  stop_reason: string | null;
  stop_rationale: string | null;
}
export interface InformationNeedResolution {
  graph_name: string;
  information_need_count: number;
  supported_information_need_ids: string[];
  supported_information_need_count: number;
  unresolved_information_need_ids: string[];
  unresolved_information_need_count: number;
  complete: boolean;
  total_retrieval_attempts: number;
  max_total_retrieval_attempts: number;
  max_attempts_per_information_need: number;
  executions: InformationNeedExecution[];
}
