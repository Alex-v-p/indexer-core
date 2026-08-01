export type DocumentUploadMode = 'automatic' | 'new_document' | 'manual_version';

export interface SubjectClassificationStatus {
  status: 'queued' | 'running' | 'succeeded' | 'failed' | string;
  job_id: string | null;
  document_version_id: string | null;
  policy_version: string | null;
  classifier_version: string | null;
  assigned_count: number | null;
  suggested_count: number | null;
  review_required_count: number | null;
  error_message: string | null;
}

export interface QueuedSubjectClassificationJob {
  job_id: string;
  status: string;
  document_id: string;
  document_version_id: string;
}

export interface QueuedSubjectClassificationResponse {
  jobs: QueuedSubjectClassificationJob[];
  skipped_document_ids: string[];
}

export interface DocumentUploadRequest {
  files: File[];
  title?: string;
  publishedAt?: string;
  detectExistingVersions: boolean;
  versionOfDocumentId?: string;
  subjectIds: string[];
}

export interface DocumentSummary {
  id: string;
  title: string;
  original_filename: string | null;
  content_type: string | null;
  storage_uri: string | null;
  size_bytes: number | null;
  checksum_sha256: string | null;
  status: string;
  chunk_count: number;
  subject_classification?: SubjectClassificationStatus | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface DocumentVersion {
  id: string;
  version_number: number;
  is_latest: boolean;
  uploaded_at: string;
  published_at: string | null;
  storage_uri: string | null;
  content_type: string | null;
  checksum_sha256: string | null;
  parser_name: string | null;
  parser_version: string | null;
  status: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ChunkIndex {
  id: string;
  ordinal: number;
  content_hash: string | null;
  token_count: number | null;
  source_page_start: number | null;
  source_page_end: number | null;
  section_title: string | null;
  qdrant_collection: string;
  qdrant_point_id: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface DocumentDetail extends DocumentSummary {
  versions: DocumentVersion[];
  chunks: ChunkIndex[];
}

export interface QueuedDocumentUpload {
  filename: string;
  document: DocumentDetail;
  jobId: string;
}

export interface RejectedDocumentUpload {
  filename: string;
  detail: string;
}

export interface BatchQueuedDocumentUpload {
  accepted: QueuedDocumentUpload[];
  rejected: RejectedDocumentUpload[];
}

export interface DocumentVersionDeletionTarget {
  documentId: string;
  versionId: string;
}
