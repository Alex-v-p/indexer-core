import {
  DocumentPreference,
  EvidenceItem,
  InformationNeed,
  InformationNeedAttemptEvidence,
  RetrievalPlan,
} from './query.models';

export type TraceEvidenceLookupItem = EvidenceItem | InformationNeedAttemptEvidence;

export interface AgentJourneyTransition {
  fromAttemptNumber: number;
  toAttemptNumber: number;
  label: string;
  reason: string | null;
  adjustments: string[];
  strategyChanged: boolean;
  pipelineChanged: boolean;
}

export interface AgentJourneyAttempt {
  attemptNumber: number;
  strategy: string;
  pipeline: string;
  query: string;
  topK: number;
  retrievedCount: number;
  uniqueEvidenceAdded: number;
  gradingStatus: string;
  coverageScore: number | null;
  gradingRationale: string | null;
  planRationale: string | null;
  adjustments: string[];
  preferredDocument: DocumentPreference | null;
  transitionFromPrevious: AgentJourneyTransition | null;
  historicalPlanOnly: boolean;
}

export interface AgentJourneyNeedLane {
  need: InformationNeed;
  status: string;
  coverageScore: number | null;
  finalRationale: string | null;
  stopReason: string | null;
  stopRationale: string | null;
  attempts: AgentJourneyAttempt[];
}

export interface StrategyEvolutionLane {
  informationNeedId: string;
  description: string;
  attempts: AgentJourneyAttempt[];
}

export interface AgentJourneySummary {
  strategiesUsed: string[];
  pipelinesUsed: string[];
  strategyChanges: number;
  pipelineChanges: number;
  totalAttempts: number;
  retries: number;
  mostUsedStrategy: string | null;
}

export interface AgentTraceViewModel {
  classificationLabel: string;
  queryStatus: string;
  informationNeedCount: number;
  resolutionStatus: string;
  hasFinalArbitration: boolean;
  initialQueryLevelPlan: RetrievalPlan | null;
  needLanes: AgentJourneyNeedLane[];
  strategyEvolution: StrategyEvolutionLane[];
  summary: AgentJourneySummary;
  informationNeedsById: ReadonlyMap<string, InformationNeed>;
  evidenceByRank: ReadonlyMap<number, TraceEvidenceLookupItem>;
  evidenceByKey: ReadonlyMap<string, TraceEvidenceLookupItem>;
  sourceLabelsByEvidenceKey: ReadonlyMap<string, string>;
}
