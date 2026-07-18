export interface QueryRequest {
  question: string;
  top_k: number;
  pipeline_name: string | null;
}

export interface ToolSummary {
  name: string;
  kind: string;
  version: string;
  description: string;
  metadata: Record<string, unknown>;
}

export interface PipelineSummary {
  name: string;
  version: string;
  description: string;
  is_default: boolean;
  tools: ToolSummary[];
  metadata: Record<string, unknown>;
}

export interface PipelineListResponse {
  default_pipeline_name: string;
  pipelines: PipelineSummary[];
}

export interface EvidenceItem {
  id: string | null;
  rank: number;
  score: number | null;
  text: string;
  qdrant_chunk_index_id: string | null;
  document_id: string | null;
  document_version_id: string | null;
  metadata: Record<string, unknown>;
}

export interface CitationItem {
  id: string | null;
  citation_index: number;
  label: string | null;
  evidence_id: string | null;
  page_number: number | null;
  quote: string | null;
  qdrant_chunk_index_id: string | null;
  document_id: string | null;
  document_version_id: string | null;
  metadata: Record<string, unknown>;
}

export interface TraceStep {
  id: string | null;
  step_order: number;
  name: string;
  step_type: string | null;
  status: string;
  duration_ms: number | null;
  input_summary: string | null;
  output_summary: string | null;
  error_message: string | null;
  metadata: Record<string, unknown>;
}

export interface VersionConstraint {
  mode: 'all' | 'latest' | 'specific' | 'previous' | 'latest_and_previous';
  version_numbers: number[];
  active: boolean;
  confidence: number;
  rationale: string;
  detector_name: string;
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
  strategy: 'baseline' | 'hybrid' | 'contextual' | 'multi_query' | 'rerank';
  selected_pipeline_name: string;
  rationale: string;
  planner_name: string;
  based_on_query_type: QueryClassification['query_type'];
  metadata_filter_hints: string[];
  requires_reranking: boolean;
  target_information_need_ids: string[];
  target_information_need_count: number;
  version_constraint: VersionConstraint;
  date_constraints: DateConstraint[];
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
  version_constraint: VersionConstraint;
  date_constraints: DateConstraint[];
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
  plan: InformationNeedRetrievalPlan;
  constraint_validation: ConstraintValidation;
  evidence_grading: Record<string, unknown>;
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

export interface QueryResponse {
  id: string;
  question: string;
  answer: string | null;
  status: string;
  pipeline_name: string | null;
  pipeline_version: string | null;
  top_k: number | null;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  classification: QueryClassification | null;
  information_need_decomposition: InformationNeedDecomposition | null;
  retrieval_plan: RetrievalPlan | null;
  information_need_resolution: InformationNeedResolution | null;
  constraint_validation: ConstraintValidation | null;
  evidence_context: EvidenceContext | null;
  evidence: EvidenceItem[];
  citations: CitationItem[];
  trace: TraceStep[];
}
