import type { BackgroundJob } from '../../../core/background-jobs/background-job.models';
import type { AnswerPresentation } from './answer.models';
import type {
  ConstraintValidation,
  DocumentPreference,
  EvidenceContext,
  EvidenceGrading,
  InformationNeedDecomposition,
  InformationNeedResolution,
  QueryClassification,
  RetrievalPlan,
  RetrievalRetry,
} from './retrieval.models';
import type { TraceStep } from './trace.models';

export interface QueryRequest {
  question: string;
  pipeline_name: string | null;
  scheduled_at?: string | null;
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
export interface QueryResponse {
  id: string;
  question: string;
  answer: string | null;
  answer_presentation?: AnswerPresentation | null;
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
  evidence_grading: EvidenceGrading | null;
  retrieval_retry: RetrievalRetry | null;
  primary_document_preference: DocumentPreference | null;
  information_need_resolution: InformationNeedResolution | null;
  constraint_validation: ConstraintValidation | null;
  evidence_context: EvidenceContext | null;
  evidence: EvidenceItem[];
  citations: CitationItem[];
  trace: TraceStep[];
}

export interface QueuedQueryResponse {
  query: QueryResponse;
  job: BackgroundJob;
}
