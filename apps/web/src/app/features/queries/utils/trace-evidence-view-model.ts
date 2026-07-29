import {
  CitationItem,
  DocumentPreference,
  EvidenceGrade,
  EvidenceGrading,
  EvidenceItem,
  InformationNeed,
  InformationNeedAttemptEvidence,
} from '../models/query.models';
import {
  TraceEvidenceCardViewModel,
  TraceEvidenceGradeView,
  TraceEvidenceNeedReference,
  TraceEvidenceSourceView,
} from '../models/trace-evidence-view.models';

interface AttemptEvidenceCardOptions {
  informationNeeds?: InformationNeed[];
  preferredDocument?: DocumentPreference | null;
  retainedInAggregate?: boolean | null;
}

interface FinalEvidenceCardOptions {
  grading: EvidenceGrading;
  evidence: EvidenceItem[];
  attemptEvidence?: InformationNeedAttemptEvidence[];
  informationNeeds?: InformationNeed[];
  citations?: CitationItem[];
}

export function buildAttemptEvidenceCard(
  evidence: InformationNeedAttemptEvidence,
  options: AttemptEvidenceCardOptions = {},
): TraceEvidenceCardViewModel {
  return {
    key: evidence.evidence_key,
    text: evidence.text,
    preview: evidencePreview(evidence.text),
    aggregateRank: evidence.aggregate_rank,
    retrievalOrder: evidence.retrieval_order,
    retrievalScore: evidence.score,
    source: evidenceSource(evidence),
    preferredDocument: evidenceMatchesPreference(evidence, options.preferredDocument ?? null),
    attemptGrade:
      evidence.relevant === null
        ? null
        : {
            relevant: evidence.relevant,
            relevanceScore: evidence.relevance_score,
            rationale: evidence.grading_rationale,
          },
    finalGrade: null,
    supportedNeeds: resolveNeedReferences(
      evidence.supports_information_need_ids ?? [],
      options.informationNeeds ?? [],
    ),
    retainedAfterNeedGrading: evidence.retained_after_need_grading,
    retainedInAggregate: options.retainedInAggregate ?? null,
    usedByAnswer: null,
    citations: [],
  };
}

export function buildFinalEvidenceCards(
  options: FinalEvidenceCardOptions,
): TraceEvidenceCardViewModel[] {
  const needs = options.informationNeeds ?? [];
  const attemptEvidence = options.attemptEvidence ?? [];
  const citations = options.citations ?? [];
  const finalByRank = new Map(options.evidence.map((item) => [item.rank, item]));
  const attemptByRank = new Map<number, InformationNeedAttemptEvidence>();
  for (const item of attemptEvidence) {
    if (item.aggregate_rank !== null && !attemptByRank.has(item.aggregate_rank)) {
      attemptByRank.set(item.aggregate_rank, item);
    }
  }

  const cards: TraceEvidenceCardViewModel[] = [];
  const renderedRanks = new Set<number>();
  const orderedGrades = [...(options.grading.grades ?? [])].sort(
    (left, right) => left.evidence_rank - right.evidence_rank,
  );
  for (const grade of orderedGrades) {
    const item = finalByRank.get(grade.evidence_rank) ?? attemptByRank.get(grade.evidence_rank);
    cards.push(finalEvidenceCard(item, grade, needs, citations, finalByRank.has(grade.evidence_rank)));
    renderedRanks.add(grade.evidence_rank);
  }

  for (const item of [...options.evidence].sort((left, right) => left.rank - right.rank)) {
    if (renderedRanks.has(item.rank)) {
      continue;
    }
    cards.push(finalEvidenceCard(item, null, needs, citations, true));
  }

  return cards;
}

export function placeholderEvidenceCard(
  rank: number,
  grade: EvidenceGrade | null,
  informationNeeds: InformationNeed[],
): TraceEvidenceCardViewModel {
  return finalEvidenceCard(undefined, grade ?? fallbackGrade(rank), informationNeeds, [], false);
}

export function evidenceSource(
  evidence: EvidenceItem | InformationNeedAttemptEvidence,
): TraceEvidenceSourceView {
  const metadata = evidence.metadata ?? {};
  const title =
    firstString(metadata, 'document_title', 'original_filename', 'filename', 'source_name') ??
    (evidence.document_id ? `Document ${evidence.document_id}` : 'Retrieved source chunk');
  const version =
    firstString(metadata, 'document_version_label', 'version_label') ??
    prefixedNumber(metadata, 'document_version_number', 'version_number', 'v');
  const pageStart = firstNumber(metadata, 'page_number', 'source_page_start');
  const pageEnd = firstNumber(metadata, 'source_page_end');
  const page =
    pageStart === null
      ? null
      : pageEnd !== null && pageEnd !== pageStart
        ? `Pages ${pageStart}–${pageEnd}`
        : `Page ${pageStart}`;
  const section = firstString(metadata, 'section_title', 'section');
  const chunkValue =
    firstNumber(metadata, 'chunk_ordinal', 'chunk_index', 'ordinal') ??
    firstString(metadata, 'chunk_ordinal', 'chunk_index', 'ordinal');

  return {
    title,
    version,
    page,
    section,
    chunk: chunkValue === null ? null : `Chunk ${chunkValue}`,
  };
}

export function evidencePreview(text: string, maxLength = 180): string {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return 'Evidence text is unavailable for this run.';
  }
  return normalized.length > maxLength
    ? `${normalized.slice(0, maxLength - 1).trimEnd()}…`
    : normalized;
}

export function resolveNeedReferences(
  needIds: string[],
  informationNeeds: InformationNeed[],
): TraceEvidenceNeedReference[] {
  const needsById = new Map(informationNeeds.map((need) => [need.need_id, need]));
  return needIds.map((needId) => ({
    needId,
    description: needsById.get(needId)?.description ?? `Unknown information need (${needId})`,
  }));
}

function finalEvidenceCard(
  evidence: EvidenceItem | InformationNeedAttemptEvidence | undefined,
  grade: EvidenceGrade | null,
  informationNeeds: InformationNeed[],
  citations: CitationItem[],
  usedByAnswer: boolean,
): TraceEvidenceCardViewModel {
  const aggregateRank =
    evidence && 'rank' in evidence
      ? evidence.rank
      : evidence?.aggregate_rank ?? grade?.evidence_rank ?? null;
  const text = evidence?.text ?? 'Evidence text is unavailable for this older run.';
  const evidenceCitations = evidence ? citations.filter((citation) => citationMatches(citation, evidence)) : [];
  const attemptGrade = evidence && 'relevant' in evidence && evidence.relevant !== null
    ? {
        relevant: evidence.relevant,
        relevanceScore: evidence.relevance_score,
        rationale: evidence.grading_rationale,
      }
    : null;

  return {
    key: evidenceKey(evidence, aggregateRank),
    text,
    preview: evidencePreview(text),
    aggregateRank,
    retrievalOrder: evidence && 'retrieval_order' in evidence ? evidence.retrieval_order : null,
    retrievalScore: evidence?.score ?? null,
    source: evidence
      ? evidenceSource(evidence)
      : {
          title: 'Source unavailable for this older run',
          version: null,
          page: null,
          section: null,
          chunk: null,
        },
    preferredDocument: null,
    attemptGrade,
    finalGrade: grade ? gradeView(grade) : null,
    supportedNeeds: resolveNeedReferences(
      grade?.supports_information_need_ids ?? [],
      informationNeeds,
    ),
    retainedAfterNeedGrading:
      evidence && 'retained_after_need_grading' in evidence
        ? evidence.retained_after_need_grading
        : null,
    retainedInAggregate: grade !== null,
    usedByAnswer,
    citations: evidenceCitations,
  };
}

function gradeView(grade: EvidenceGrade): TraceEvidenceGradeView {
  return {
    relevant: grade.relevant,
    relevanceScore: grade.relevance_score,
    rationale: grade.rationale,
  };
}

function evidenceMatchesPreference(
  evidence: InformationNeedAttemptEvidence,
  preference: DocumentPreference | null,
): boolean | null {
  const balancing = record(evidence.metadata?.['document_balancing']);
  if (typeof balancing?.['primary_document'] === 'boolean') {
    return balancing['primary_document'];
  }
  if (!preference) {
    return null;
  }

  const reference = preference.document;
  if (reference.document_id && evidence.document_id === reference.document_id) {
    return true;
  }
  if (
    evidence.document_version_id &&
    (reference.document_version_ids ?? []).includes(evidence.document_version_id)
  ) {
    return true;
  }

  const evidenceNames = [
    firstString(evidence.metadata ?? {}, 'document_title', 'original_filename', 'filename'),
  ]
    .filter((value): value is string => value !== null)
    .map(normalizeName);
  const referenceNames = [
    reference.display_name,
    ...(reference.normalized_names ?? []),
  ].map(normalizeName);
  return evidenceNames.some((name) => referenceNames.includes(name));
}

function citationMatches(
  citation: CitationItem,
  evidence: EvidenceItem | InformationNeedAttemptEvidence,
): boolean {
  const ids = 'id' in evidence
    ? [evidence.id, evidence.qdrant_chunk_index_id]
    : [evidence.evidence_key, evidence.qdrant_chunk_index_id];
  if (citation.evidence_id && ids.includes(citation.evidence_id)) {
    return true;
  }
  if (
    citation.qdrant_chunk_index_id &&
    citation.qdrant_chunk_index_id === evidence.qdrant_chunk_index_id
  ) {
    return true;
  }
  return false;
}

function evidenceKey(
  evidence: EvidenceItem | InformationNeedAttemptEvidence | undefined,
  rank: number | null,
): string {
  if (evidence && 'evidence_key' in evidence) {
    return evidence.evidence_key;
  }
  if (evidence?.id) {
    return evidence.id;
  }
  if (evidence?.qdrant_chunk_index_id) {
    return `qdrant_chunk:${evidence.qdrant_chunk_index_id}`;
  }
  return `aggregate-rank:${rank ?? 'unknown'}`;
}

function fallbackGrade(rank: number): EvidenceGrade {
  return {
    evidence_rank: rank,
    relevance_score: 0,
    relevant: false,
    rationale: 'No evidence details were stored for this older run.',
    supports_information_need_ids: [],
  };
}

function firstString(metadata: Record<string, unknown>, ...keys: string[]): string | null {
  for (const key of keys) {
    const value = metadata[key];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function firstNumber(metadata: Record<string, unknown>, ...keys: string[]): number | null {
  for (const key of keys) {
    const value = metadata[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function prefixedNumber(
  metadata: Record<string, unknown>,
  firstKey: string,
  secondKey: string,
  prefix: string,
): string | null {
  const value = firstNumber(metadata, firstKey, secondKey);
  return value === null ? null : `${prefix}${value}`;
}

function normalizeName(value: string): string {
  return value.toLocaleLowerCase().replace(/\.[a-z0-9]+$/i, '').replace(/[^a-z0-9]+/g, '');
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}
