import {
  AgentJourneyAttempt,
  AgentJourneyNeedLane,
  AgentJourneySummary,
  AgentJourneyTransition,
  AgentTraceViewModel,
  TraceEvidenceLookupItem,
} from '../view-models/agent-trace-view.models';
import {
  EvidenceItem,
  InformationNeed,
  InformationNeedAttempt,
  InformationNeedAttemptEvidence,
  InformationNeedExecution,
  InformationNeedRetrievalPlan,
  QueryResponse,
} from '../models/query.models';

export function buildAgentTraceViewModel(result: QueryResponse): AgentTraceViewModel {
  const informationNeedsById = buildInformationNeedLookup(result);
  const needLanes = buildNeedLanes(result, informationNeedsById);
  const evidenceLookups = buildEvidenceLookups(result);
  const summary = buildSummary(result, needLanes);

  return {
    classificationLabel: displayLabel(result.classification?.query_type ?? 'not_classified'),
    queryStatus: displayLabel(result.status || 'unknown'),
    informationNeedCount:
      result.information_need_decomposition?.information_need_count ??
      result.information_need_resolution?.information_need_count ??
      needLanes.length,
    resolutionStatus: result.information_need_resolution
      ? result.information_need_resolution.complete
        ? 'Complete'
        : 'Partial'
      : 'Unavailable',
    hasFinalArbitration: result.evidence_grading !== null,
    initialQueryLevelPlan: result.retrieval_plan ?? null,
    needLanes,
    strategyEvolution: needLanes.map((lane) => ({
      informationNeedId: lane.need.need_id,
      description: lane.need.description,
      attempts: lane.attempts,
    })),
    summary,
    informationNeedsById,
    evidenceByRank: evidenceLookups.byRank,
    evidenceByKey: evidenceLookups.byKey,
    sourceLabelsByEvidenceKey: evidenceLookups.sourceLabels,
  };
}

export function displayLabel(value: string | null | undefined): string {
  if (!value) {
    return 'Unavailable';
  }
  return value.replaceAll('_', ' ');
}

export function formatMetadataValue(value: unknown): string | null {
  if (typeof value === 'string') {
    return value.trim() || null;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value)) {
    const values = value
      .map((item) => formatMetadataValue(item))
      .filter((item): item is string => item !== null);
    return values.length > 0 ? values.join(', ') : null;
  }
  return null;
}

export function sourceDisplayLabel(evidence: TraceEvidenceLookupItem): string {
  const metadata = evidence.metadata ?? {};
  for (const key of [
    'document_title',
    'title',
    'original_filename',
    'filename',
    'source_name',
  ]) {
    const label = formatMetadataValue(metadata[key]);
    if (label) {
      return label;
    }
  }

  if (evidence.document_id) {
    return `Document ${evidence.document_id}`;
  }
  if (isAttemptEvidence(evidence)) {
    return `Retrieved item ${evidence.retrieval_order}`;
  }
  return `Evidence ${evidence.rank}`;
}

function buildInformationNeedLookup(result: QueryResponse): ReadonlyMap<string, InformationNeed> {
  const lookup = new Map<string, InformationNeed>();
  for (const need of result.information_need_decomposition?.information_needs ?? []) {
    lookup.set(need.need_id, need);
  }
  for (const execution of result.information_need_resolution?.executions ?? []) {
    lookup.set(execution.information_need_id, execution.information_need);
  }
  return lookup;
}

function buildNeedLanes(
  result: QueryResponse,
  informationNeedsById: ReadonlyMap<string, InformationNeed>,
): AgentJourneyNeedLane[] {
  const executions = result.information_need_resolution?.executions ?? [];
  const executionsById = new Map(
    executions.map((execution) => [execution.information_need_id, execution]),
  );
  const orderedNeeds = [...informationNeedsById.values()];

  if (orderedNeeds.length === 0) {
    return executions.map((execution) => buildNeedLane(execution.information_need, execution));
  }

  return orderedNeeds.map((need) => buildNeedLane(need, executionsById.get(need.need_id)));
}

function buildNeedLane(
  need: InformationNeed,
  execution: InformationNeedExecution | undefined,
): AgentJourneyNeedLane {
  const attempts = buildAttempts(execution);
  return {
    need,
    status: execution?.status ?? 'pending',
    coverageScore: execution?.final_grade?.coverage_score ?? null,
    finalRationale: execution?.final_grade?.rationale ?? null,
    stopReason: execution?.stop_reason ?? null,
    stopRationale: execution?.stop_rationale ?? null,
    attempts,
  };
}

function buildAttempts(execution: InformationNeedExecution | undefined): AgentJourneyAttempt[] {
  if (!execution) {
    return [];
  }

  const orderedAttempts = [...(execution.attempts ?? [])].sort(
    (left, right) => left.attempt_number - right.attempt_number,
  );
  if (orderedAttempts.length > 0) {
    return orderedAttempts.map((attempt, index) =>
      buildAttempt(
        attempt,
        index > 0 ? orderedAttempts[index - 1] : null,
        execution,
      ),
    );
  }

  const orderedPlans = [...(execution.plan_history ?? [])].sort(
    (left, right) => left.attempt_number - right.attempt_number,
  );
  return orderedPlans.map((plan, index) =>
    buildPlanOnlyAttempt(plan, index > 0 ? orderedPlans[index - 1] : null, execution),
  );
}

function buildAttempt(
  attempt: InformationNeedAttempt,
  previous: InformationNeedAttempt | null,
  execution: InformationNeedExecution,
): AgentJourneyAttempt {
  return {
    attemptNumber: attempt.attempt_number,
    strategy: attempt.strategy || attempt.plan?.strategy || 'unknown',
    pipeline: attempt.pipeline_name || attempt.plan?.selected_pipeline_name || 'unknown',
    query: attempt.query || attempt.plan?.query || execution.information_need.retrieval_query,
    topK: attempt.top_k || attempt.plan?.top_k || 0,
    retrievedCount: attempt.retrieved_count ?? 0,
    uniqueEvidenceAdded: attempt.unique_evidence_added ?? 0,
    gradingStatus: attempt.evidence_grading?.status ?? 'not_graded',
    coverageScore: attempt.evidence_grading?.coverage_score ?? null,
    gradingRationale: attempt.evidence_grading?.rationale ?? null,
    planRationale: attempt.plan?.rationale ?? null,
    adjustments: attempt.adjustments ?? attempt.plan?.adjustments ?? [],
    preferredDocument: attempt.plan?.preferred_document ?? null,
    transitionFromPrevious: previous
      ? buildTransition(
          toComparableAttempt(previous),
          toComparableAttempt(attempt),
          attempt.evidence_grading?.rationale ?? null,
        )
      : null,
    historicalPlanOnly: false,
  };
}

function buildPlanOnlyAttempt(
  plan: InformationNeedRetrievalPlan,
  previous: InformationNeedRetrievalPlan | null,
  execution: InformationNeedExecution,
): AgentJourneyAttempt {
  return {
    attemptNumber: plan.attempt_number,
    strategy: plan.strategy,
    pipeline: plan.selected_pipeline_name,
    query: plan.query,
    topK: plan.top_k,
    retrievedCount: 0,
    uniqueEvidenceAdded: 0,
    gradingStatus: execution.final_grade?.status ?? 'not_graded',
    coverageScore: execution.final_grade?.coverage_score ?? null,
    gradingRationale: execution.final_grade?.rationale ?? null,
    planRationale: plan.rationale,
    adjustments: plan.adjustments ?? [],
    preferredDocument: plan.preferred_document ?? null,
    transitionFromPrevious: previous
      ? buildTransition(
          toComparablePlan(previous),
          toComparablePlan(plan),
          execution.final_grade?.rationale ?? null,
        )
      : null,
    historicalPlanOnly: true,
  };
}

interface ComparableAttempt {
  attemptNumber: number;
  strategy: string;
  pipeline: string;
  query: string;
  topK: number;
  adjustments: string[];
  gradingStatus: string;
  gradingRationale: string | null;
}

function toComparableAttempt(attempt: InformationNeedAttempt): ComparableAttempt {
  return {
    attemptNumber: attempt.attempt_number,
    strategy: attempt.strategy,
    pipeline: attempt.pipeline_name,
    query: attempt.query,
    topK: attempt.top_k,
    adjustments: attempt.adjustments ?? [],
    gradingStatus: attempt.evidence_grading?.status ?? 'not_graded',
    gradingRationale: attempt.evidence_grading?.rationale ?? null,
  };
}

function toComparablePlan(plan: InformationNeedRetrievalPlan): ComparableAttempt {
  return {
    attemptNumber: plan.attempt_number,
    strategy: plan.strategy,
    pipeline: plan.selected_pipeline_name,
    query: plan.query,
    topK: plan.top_k,
    adjustments: plan.adjustments ?? [],
    gradingStatus: 'not_graded',
    gradingRationale: null,
  };
}

function buildTransition(
  previous: ComparableAttempt,
  current: ComparableAttempt,
  fallbackReason: string | null,
): AgentJourneyTransition {
  const strategyChanged = previous.strategy !== current.strategy;
  const pipelineChanged = previous.pipeline !== current.pipeline;
  const parts: string[] = [];

  if (previous.gradingStatus === 'weak' || previous.gradingStatus === 'missing') {
    parts.push(`${displayLabel(previous.gradingStatus)} evidence`);
  }
  for (const adjustment of current.adjustments) {
    parts.push(displayLabel(adjustment));
  }
  if (strategyChanged) {
    parts.push(`switched ${displayLabel(previous.strategy)} to ${displayLabel(current.strategy)}`);
  }
  if (pipelineChanged) {
    parts.push(`changed pipeline to ${current.pipeline}`);
  }
  if (current.topK > previous.topK && !parts.some((part) => part.includes('top k'))) {
    parts.push(`increased top-k to ${current.topK}`);
  }
  if (current.query !== previous.query && current.adjustments.length === 0) {
    parts.push('refined query');
  }

  return {
    fromAttemptNumber: previous.attemptNumber,
    toAttemptNumber: current.attemptNumber,
    label: parts.length > 0 ? parts.join(' → ') : 'Retry with revised retrieval plan',
    reason: previous.gradingRationale ?? fallbackReason,
    adjustments: [...current.adjustments],
    strategyChanged,
    pipelineChanged,
  };
}

function buildSummary(
  result: QueryResponse,
  lanes: AgentJourneyNeedLane[],
): AgentJourneySummary {
  const attempts = lanes.flatMap((lane) => lane.attempts);
  const strategiesUsed = unique(attempts.map((attempt) => attempt.strategy).filter(isKnownValue));
  const pipelinesUsed = unique(attempts.map((attempt) => attempt.pipeline).filter(isKnownValue));
  const transitions = attempts
    .map((attempt) => attempt.transitionFromPrevious)
    .filter((transition): transition is AgentJourneyTransition => transition !== null);
  const countedAttempts = attempts.length;
  const reportedAttempts =
    result.information_need_resolution?.total_retrieval_attempts ??
    result.retrieval_retry?.attempt_count ??
    countedAttempts;
  const retries = lanes.reduce(
    (total, lane) => total + Math.max(lane.attempts.length - 1, 0),
    0,
  );
  const reportedNeedRetries = (result.information_need_resolution?.executions ?? []).reduce(
    (total, execution) => total + Math.max(execution.attempts_used - 1, 0),
    0,
  );

  return {
    strategiesUsed,
    pipelinesUsed,
    strategyChanges: transitions.filter((transition) => transition.strategyChanged).length,
    pipelineChanges: transitions.filter((transition) => transition.pipelineChanged).length,
    totalAttempts: Math.max(countedAttempts, reportedAttempts),
    retries: Math.max(retries, reportedNeedRetries, result.retrieval_retry?.retries_used ?? 0),
    mostUsedStrategy: mostUsed(attempts.map((attempt) => attempt.strategy).filter(isKnownValue)),
  };
}

function buildEvidenceLookups(result: QueryResponse): {
  byRank: ReadonlyMap<number, TraceEvidenceLookupItem>;
  byKey: ReadonlyMap<string, TraceEvidenceLookupItem>;
  sourceLabels: ReadonlyMap<string, string>;
} {
  const byRank = new Map<number, TraceEvidenceLookupItem>();
  const byKey = new Map<string, TraceEvidenceLookupItem>();
  const sourceLabels = new Map<string, string>();

  for (const evidence of result.evidence ?? []) {
    byRank.set(evidence.rank, evidence);
    for (const key of evidenceKeys(evidence)) {
      byKey.set(key, evidence);
      sourceLabels.set(key, sourceDisplayLabel(evidence));
    }
  }

  for (const execution of result.information_need_resolution?.executions ?? []) {
    for (const attempt of execution.attempts ?? []) {
      for (const evidence of attempt.evidence ?? []) {
        if (evidence.aggregate_rank !== null && !byRank.has(evidence.aggregate_rank)) {
          byRank.set(evidence.aggregate_rank, evidence);
        }
        byKey.set(evidence.evidence_key, evidence);
        sourceLabels.set(evidence.evidence_key, sourceDisplayLabel(evidence));
      }
    }
  }

  return { byRank, byKey, sourceLabels };
}

function evidenceKeys(evidence: EvidenceItem): string[] {
  const keys = [evidence.id, evidence.qdrant_chunk_index_id].filter(
    (value): value is string => Boolean(value),
  );
  if (evidence.qdrant_chunk_index_id) {
    keys.push(`qdrant_chunk:${evidence.qdrant_chunk_index_id}`);
  }
  return keys;
}

function isAttemptEvidence(
  evidence: TraceEvidenceLookupItem,
): evidence is InformationNeedAttemptEvidence {
  return 'evidence_key' in evidence;
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function isKnownValue(value: string): boolean {
  return Boolean(value && value !== 'unknown');
}

function mostUsed(values: string[]): string | null {
  const counts = new Map<string, number>();
  let mostUsedValue: string | null = null;
  let mostUsedCount = 0;
  for (const value of values) {
    const count = (counts.get(value) ?? 0) + 1;
    counts.set(value, count);
    if (count > mostUsedCount) {
      mostUsedValue = value;
      mostUsedCount = count;
    }
  }
  return mostUsedValue;
}
