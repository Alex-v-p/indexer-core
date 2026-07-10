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
  evidence: EvidenceItem[];
  citations: CitationItem[];
  trace: TraceStep[];
}
