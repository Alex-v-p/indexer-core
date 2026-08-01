import type {
  InformationNeedExecution,
  QueryResponse,
  SubjectDocumentLane,
} from '../models/query.models';

export interface ComparisonLaneViewModel {
  subjectId: string;
  subjectName: string;
  state: string;
  support: 'supported' | 'missing' | 'unsupported';
  searchedDocumentCount: number;
  contributingDocumentCount: number;
  coverageSatisfied: boolean | null;
  rejectedCount: number;
}

export interface SubjectScopeViewModel {
  subjects: Array<{ id: string; name: string; kind: string }>;
  source: string;
  confidencePercent: number;
  strict: boolean;
  eligibleDocumentCount: number;
  requestedCoverage: string;
  effectiveCoverage: string;
  fallback: string | null;
  comparisonRequested: boolean;
  lanes: ComparisonLaneViewModel[];
}

export function buildSubjectScopeViewModel(
  result: Pick<QueryResponse, 'subject_scope' | 'information_need_resolution'>,
): SubjectScopeViewModel | null {
  const scope = result.subject_scope;
  if (!scope) {
    return null;
  }
  const catalog = scope.catalog ?? [];
  const matched = new Set(scope.matched_subject_ids ?? []);
  const executions = result.information_need_resolution?.executions ?? [];
  const coverage = executions.flatMap(latestCoverage);
  const effectiveModes = uniqueStrings(coverage.map((item) => stringValue(item, 'effective_mode')));
  const fallbackReasons = uniqueStrings(coverage.map((item) => stringValue(item, 'fallback_reason')));
  return {
    subjects: catalog
      .filter((subject) => matched.has(subject.subject_id))
      .map((subject) => ({ id: subject.subject_id, name: subject.name, kind: subject.kind })),
    source: scope.source,
    confidencePercent: Math.round(Math.min(Math.max(scope.confidence, 0), 1) * 100),
    strict: scope.strict,
    eligibleDocumentCount: scope.document_scope.allowed_document_count,
    requestedCoverage: scope.coverage_mode,
    effectiveCoverage: effectiveModes.join(', ') || scope.coverage_mode,
    fallback: fallbackReasons.join(', ') || null,
    comparisonRequested: scope.comparison_requested ?? false,
    lanes: (scope.subject_lanes ?? []).map((lane) => laneViewModel(lane, executions)),
  };
}

function laneViewModel(
  lane: SubjectDocumentLane,
  executions: InformationNeedExecution[],
): ComparisonLaneViewModel {
  const laneExecutions = executions.filter(
    (execution) => execution.information_need.subject_lane?.subject_id === lane.subject_id,
  );
  const coverage = laneExecutions.flatMap(latestCoverage);
  const searchedIds = uniqueStrings(coverage.flatMap((item) => stringArray(item, 'searched_document_ids')));
  const contributingIds = uniqueStrings(coverage.flatMap((item) => stringArray(item, 'contributing_document_ids')));
  const searchedCount = searchedIds.length || maxNumber(coverage, 'searched_document_count');
  const contributingCount = contributingIds.length || maxNumber(coverage, 'contributing_document_count');
  const finalGrades = laneExecutions.map((execution) => execution.final_grade).filter(Boolean);
  const supported = finalGrades.some((grade) => grade?.status === 'supported');
  const missing = laneExecutions.length === 0 || finalGrades.some((grade) => grade?.status === 'missing');
  const satisfaction = coverage.map((item) => booleanValue(item, 'coverage_satisfied')).filter((value): value is boolean => value !== null);
  return {
    subjectId: lane.subject_id,
    subjectName: lane.subject_name,
    state: supported ? 'supported' : laneExecutions.at(-1)?.status ?? 'not_searched',
    support: supported ? 'supported' : missing ? 'missing' : 'unsupported',
    searchedDocumentCount: searchedCount,
    contributingDocumentCount: contributingCount,
    coverageSatisfied: satisfaction.length ? satisfaction.every(Boolean) : null,
    rejectedCount: coverage.reduce((total, item) => total + numberValue(item, 'out_of_scope_rejected_count'), 0),
  };
}

function latestCoverage(execution: InformationNeedExecution): Record<string, unknown>[] {
  const coverage = execution.attempts.at(-1)?.retrieval_metadata.coverage;
  return coverage ? [coverage] : [];
}

function uniqueStrings(values: Array<string | null>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))];
}

function stringValue(value: Record<string, unknown>, key: string): string | null {
  return typeof value[key] === 'string' && value[key] ? value[key] as string : null;
}

function stringArray(value: Record<string, unknown>, key: string): string[] {
  const raw = value[key];
  return Array.isArray(raw) ? raw.filter((item): item is string => typeof item === 'string') : [];
}

function numberValue(value: Record<string, unknown>, key: string): number {
  return typeof value[key] === 'number' ? value[key] as number : 0;
}

function maxNumber(values: Record<string, unknown>[], key: string): number {
  return Math.max(0, ...values.map((value) => numberValue(value, key)));
}

function booleanValue(value: Record<string, unknown>, key: string): boolean | null {
  return typeof value[key] === 'boolean' ? value[key] as boolean : null;
}
