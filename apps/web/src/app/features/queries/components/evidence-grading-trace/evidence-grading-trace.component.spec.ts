import { TestBed } from '@angular/core/testing';

import {
  CitationItem,
  EvidenceGrading,
  EvidenceItem,
  InformationNeed,
  InformationNeedAttemptEvidence,
} from '../../models/query.models';
import { EvidenceGradingTraceComponent } from './evidence-grading-trace.component';

describe('EvidenceGradingTraceComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EvidenceGradingTraceComponent],
    }).compileComponents();
  });

  it('joins final grades to accepted and rejected evidence content by aggregate rank', () => {
    const component = new EvidenceGradingTraceComponent();
    component.grading = grading();
    component.evidence = acceptedEvidence();
    component.attemptEvidence = rejectedAttemptEvidence();
    component.informationNeeds = informationNeeds();
    component.citations = citations();

    const cards = component.evidenceCards();

    expect(cards).toHaveLength(2);
    expect(cards[0].text).toContain('rolling deployment');
    expect(cards[0].finalGrade?.relevant).toBe(true);
    expect(cards[0].usedByAnswer).toBe(true);
    expect(cards[0].citations[0].label).toBe('[1]');
    expect(cards[1].text).toContain('billing paragraph');
    expect(cards[1].finalGrade?.relevant).toBe(false);
    expect(cards[1].usedByAnswer).toBe(false);
    expect(cards[1].supportedNeeds[0].description).toBe('Explain fallback behavior');
  });

  it('groups actual evidence cards by need and preserves unresolved rationale', () => {
    const component = new EvidenceGradingTraceComponent();
    component.grading = grading();
    component.evidence = acceptedEvidence();
    component.attemptEvidence = rejectedAttemptEvidence();
    component.informationNeeds = informationNeeds();

    const views = component.informationNeedViews();

    expect(views).toHaveLength(2);
    expect(views[0].need.description).toBe('Identify deployment requirements');
    expect(views[0].supportingEvidence[0].text).toContain('rolling deployment');
    expect(views[1].need.required).toBe(false);
    expect(views[1].unresolvedExplanation).toContain('does not establish fallback behavior');
  });

  it('renders expandable full evidence, human-readable need links, rejection, and citations', () => {
    const fixture = TestBed.createComponent(EvidenceGradingTraceComponent);
    fixture.componentInstance.grading = grading();
    fixture.componentInstance.evidence = acceptedEvidence();
    fixture.componentInstance.attemptEvidence = rejectedAttemptEvidence();
    fixture.componentInstance.informationNeeds = informationNeeds();
    fixture.componentInstance.citations = citations();
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    const text = element.textContent ?? '';
    const evidenceFirst = element.querySelector<HTMLDetailsElement>('[data-testid="evidence-first-view"]');
    const needFirst = element.querySelector<HTMLDetailsElement>('[data-testid="need-first-view"]');
    const needViews = Array.from(
      element.querySelectorAll<HTMLDetailsElement>('[data-testid="arbitration-need"]'),
    );

    expect(element.querySelectorAll('details[data-testid="trace-evidence-card"]').length).toBeGreaterThanOrEqual(2);
    expect(evidenceFirst?.open).toBe(false);
    expect(needFirst?.open).toBe(true);
    expect(needViews.map((view) => view.open)).toEqual([false, true]);
    expect(text).toContain('Deployment uses a rolling deployment with health checks.');
    expect(text).toContain('This billing paragraph is unrelated to fallback behavior.');
    expect(text).toContain('Final arbitration: accepted');
    expect(text).toContain('Final arbitration: rejected');
    expect(text).toContain('Explain fallback behavior');
    expect(text).toContain('The evidence does not establish fallback behavior.');
    expect(element.querySelector('[data-testid="evidence-citations"]')?.textContent).toContain('[1]');
  });

  it('renders safely when optional evidence metadata is absent', () => {
    const fixture = TestBed.createComponent(EvidenceGradingTraceComponent);
    fixture.componentInstance.grading = {
      ...grading(),
      grades: [],
      information_need_grades: [],
      unresolved_information: [],
    };
    fixture.componentInstance.informationNeeds = informationNeeds();

    expect(() => fixture.detectChanges()).not.toThrow();
    expect(fixture.nativeElement.textContent).toContain(
      'No final arbitration grade was stored for this information need.',
    );
  });
});

function informationNeeds(): InformationNeed[] {
  return [
    {
      need_id: 'need-1',
      description: 'Identify deployment requirements',
      retrieval_query: 'deployment requirements',
      required: true,
    },
    {
      need_id: 'need-2',
      description: 'Explain fallback behavior',
      retrieval_query: 'fallback behavior',
      required: false,
    },
  ];
}

function acceptedEvidence(): EvidenceItem[] {
  return [
    {
      id: 'evidence-1',
      rank: 1,
      score: 0.93,
      text: 'Deployment uses a rolling deployment with health checks.',
      qdrant_chunk_index_id: 'accepted',
      document_id: 'document-1',
      document_version_id: 'version-2',
      metadata: {
        original_filename: 'deployment-guide.pdf',
        document_version_label: 'v2',
        page_number: 7,
        section_title: 'Rollout',
        chunk_ordinal: 12,
      },
    },
  ];
}

function rejectedAttemptEvidence(): InformationNeedAttemptEvidence[] {
  return [
    {
      evidence_key: 'qdrant_chunk:rejected',
      retrieval_order: 2,
      aggregate_rank: 2,
      text: 'This billing paragraph is unrelated to fallback behavior.',
      score: 0.41,
      qdrant_chunk_index_id: 'rejected',
      document_id: 'document-2',
      document_version_id: 'version-1',
      metadata: {
        original_filename: 'billing-guide.pdf',
        document_version_label: 'v1',
        page_number: 3,
      },
      relevance_score: 0.18,
      relevant: false,
      grading_rationale: 'Rejected during need-level grading.',
      supports_information_need_ids: [],
      retained_after_need_grading: false,
    },
  ];
}

function citations(): CitationItem[] {
  return [
    {
      id: 'citation-1',
      citation_index: 1,
      label: '[1]',
      evidence_id: 'evidence-1',
      page_number: 7,
      quote: 'rolling deployment with health checks',
      qdrant_chunk_index_id: 'accepted',
      document_id: 'document-1',
      document_version_id: 'version-2',
      metadata: {},
    },
  ];
}

function grading(): EvidenceGrading {
  return {
    status: 'weak',
    coverage_score: 0.55,
    sufficient: false,
    answerable: true,
    partial_answer_available: true,
    missing_evidence: false,
    weak_evidence: true,
    relevant_count: 1,
    total_count: 2,
    relevant_evidence_ranks: [1],
    supported_information_need_count: 1,
    supported_required_information_need_count: 1,
    required_information_need_count: 1,
    supported_information: ['Deployment requirements'],
    partial_information_need_count: 0,
    missing_information_need_count: 1,
    total_information_need_count: 2,
    unresolved_information: ['Fallback behavior remains unresolved.'],
    rationale: 'Deployment is supported, but fallback behavior is not.',
    grader_name: 'final-arbitrator',
    fallback_used: false,
    grades: [
      {
        evidence_rank: 1,
        relevance_score: 0.94,
        relevant: true,
        rationale: 'This directly supports deployment requirements.',
        supports_information_need_ids: ['need-1'],
      },
      {
        evidence_rank: 2,
        relevance_score: 0.16,
        relevant: false,
        rationale: 'This is about billing, not deployment fallback.',
        supports_information_need_ids: ['need-2'],
      },
    ],
    information_need_grades: [
      {
        information_need_id: 'need-1',
        description: 'Identify deployment requirements',
        status: 'supported',
        coverage_score: 0.94,
        supporting_evidence_ranks: [1],
        rationale: 'Evidence #1 directly resolves this need.',
        required: true,
      },
      {
        information_need_id: 'need-2',
        description: 'Explain fallback behavior',
        status: 'missing',
        coverage_score: 0.1,
        supporting_evidence_ranks: [],
        rationale: 'The evidence does not establish fallback behavior.',
        required: false,
      },
    ],
  };
}
