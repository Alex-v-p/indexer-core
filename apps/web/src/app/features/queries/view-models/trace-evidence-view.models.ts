import type { CitationItem } from '../models/query-contract.models';

export interface TraceEvidenceNeedReference {
  needId: string;
  description: string;
}

export interface TraceEvidenceGradeView {
  relevant: boolean;
  relevanceScore: number | null;
  rationale: string | null;
}

export interface TraceEvidenceSourceView {
  title: string;
  version: string | null;
  page: string | null;
  section: string | null;
  chunk: string | null;
}

export interface TraceEvidenceCardViewModel {
  key: string;
  text: string;
  preview: string;
  aggregateRank: number | null;
  retrievalOrder: number | null;
  retrievalScore: number | null;
  source: TraceEvidenceSourceView;
  preferredDocument: boolean | null;
  attemptGrade: TraceEvidenceGradeView | null;
  finalGrade: TraceEvidenceGradeView | null;
  supportedNeeds: TraceEvidenceNeedReference[];
  retainedAfterNeedGrading: boolean | null;
  retainedInAggregate: boolean | null;
  usedByAnswer: boolean | null;
  citations: CitationItem[];
}
