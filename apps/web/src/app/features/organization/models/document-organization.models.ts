export type OrganizationSource = 'manual' | 'automatic';
export type ConfidenceBand = 'low' | 'medium' | 'high';
export type DocumentTypeDecisionState = 'suggested' | 'assigned' | 'rejected';
export type ContentGroupAssignmentState = 'pending' | 'unresolved' | 'suggested' | 'assigned';

export interface DocumentType {
  id: string;
  key: string;
  label: string;
  description: string | null;
  metadata: Record<string, unknown>;
  archived_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ContentGroupAlias {
  id: string;
  content_group_id: string;
  name: string;
  normalized_name: string;
  archived_at: string | null;
  created_at: string;
}

export interface ContentGroup {
  id: string;
  name: string;
  normalized_name: string;
  description: string | null;
  metadata: Record<string, unknown>;
  archived_at: string | null;
  created_at: string;
  updated_at: string;
  aliases: ContentGroupAlias[];
}

export interface DocumentTypeDecision {
  id: string;
  document_id: string;
  document_type_id: string;
  state: DocumentTypeDecisionState;
  source: OrganizationSource;
  confidence: number | null;
  confidence_band: ConfidenceBand | null;
  rationale: string | null;
  classifier_version: string | null;
  policy_version: string | null;
  signals: Record<string, unknown>;
  classified_document_version_id: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface DocumentTypeDecisionView {
  decision: DocumentTypeDecision;
  document_type: DocumentType;
}

export interface DocumentContentGroupAssignment {
  document_id: string;
  content_group_id: string | null;
  state: ContentGroupAssignmentState;
  source: OrganizationSource;
  unresolved_reason: string | null;
  confidence: number | null;
  confidence_band: ConfidenceBand | null;
  rationale: string | null;
  classifier_version: string | null;
  policy_version: string | null;
  signals: Record<string, unknown>;
  summary_hash: string | null;
  classified_document_version_id: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface DocumentOrganization {
  document_id: string;
  type_decisions: DocumentTypeDecisionView[];
  content_group_assignment: DocumentContentGroupAssignment | null;
  content_group: ContentGroup | null;
  status: Record<string, unknown> | null;
}

export interface ManualDocumentTypeDecision {
  document_type_id: string;
  state: 'assigned' | 'rejected';
  expected_revision: number;
  rationale?: string;
}

export interface OrganizationEnqueueResponse {
  job_ids: string[];
  skipped_document_ids: string[];
}
