export type SubjectKind = 'project' | 'topic' | 'organization' | 'custom';
export type SubjectDecisionState = 'suggested' | 'assigned' | 'rejected';
export type SubjectDecisionSource = 'manual' | 'automatic';
export type SubjectConfidenceBand = 'low' | 'medium' | 'high';

export interface SubjectAlias {
  id: string;
  subject_id: string;
  name: string;
  normalized_name: string;
  created_at: string;
  archived_at: string | null;
}

export interface Subject {
  id: string;
  kind: SubjectKind;
  name: string;
  normalized_name: string;
  description: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
  aliases: SubjectAlias[];
}

export interface DocumentSubjectDecision {
  id: string;
  document_id: string;
  subject_id: string;
  state: SubjectDecisionState;
  control_source: SubjectDecisionSource;
  confidence: number | null;
  confidence_band: SubjectConfidenceBand | null;
  rationale: string | null;
  classifier_version: string | null;
  policy_version: string | null;
  signals: Record<string, unknown>;
  classified_document_version_id: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface SubjectDecisionConflictDetail {
  message: string;
  current: DocumentSubjectDecision | null;
}
