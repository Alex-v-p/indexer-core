import { DocumentPreferenceTraceComponent } from './document-preference-trace.component';
import { TraceStep } from '../../models/query.models';

describe('DocumentPreferenceTraceComponent', () => {
  it('extracts primary-document detection and balancing decisions from runtime trace metadata', () => {
    const component = new DocumentPreferenceTraceComponent();
    component.trace = buildTrace();

    const detections = component.detectionEvents();
    const balancing = component.balancingEvents();

    expect(detections).toHaveLength(1);
    expect(detections[0].preference?.document.display_name).toBe('FunctionalSpecDAF_AVP.pdf');
    expect(detections[0].detection.preference_changed).toBe(true);
    expect(balancing).toHaveLength(1);
    expect(balancing[0].primaryDocument).toBe('FunctionalSpecDAF_AVP.pdf');
    expect(balancing[0].balancing.primary_selected_count).toBe(3);
    expect(component.selectedDocumentCounts(balancing[0])).toEqual([
      { key: 'document:functional', count: 3 },
      { key: 'document:realization', count: 2 },
    ]);
  });
});

function buildTrace(): TraceStep[] {
  return [
    {
      id: null,
      step_order: 8,
      name: 'detect_primary_document',
      step_type: 'document_preference',
      status: 'completed',
      duration_ms: 2,
      input_summary: null,
      output_summary: null,
      error_message: null,
      metadata: {
        primary_document_detection: {
          information_need_id: 'need_1',
          detector_name: 'rule_based_soft_primary_document_detector',
          candidate_evidence_count: 5,
          previous_document_key: null,
          selected_document_key: 'document:functional',
          preference_changed: true,
        },
        primary_document_preference: {
          document: {
            key: 'document:functional',
            display_name: 'FunctionalSpecDAF_AVP.pdf',
            document_id: null,
            document_version_ids: [],
            normalized_names: ['functionalspecdaf avp'],
          },
          score: 0.9,
          confidence: 0.86,
          margin: 0.2,
          supporting_information_need_ids: ['need_1'],
          supporting_evidence_ranks: [1, 2],
          rationale: 'The functional specification was the strongest directly graded document.',
          detector_name: 'rule_based_soft_primary_document_detector',
          semantics: 'soft_preference_not_filter',
        },
      },
    },
    {
      id: null,
      step_order: 12,
      name: 'execute_information_need_plan',
      step_type: 'information_need_retrieval',
      status: 'completed',
      duration_ms: 12,
      input_summary: null,
      output_summary: null,
      error_message: null,
      metadata: {
        active_information_need_lookup: {
          information_need_id: 'need_2',
          attempt_number: 1,
          primary_document: 'FunctionalSpecDAF_AVP.pdf',
          document_balancing: {
            selector_name: 'document_balanced_candidate_selector',
            candidate_count: 15,
            requested_top_k: 5,
            selected_count: 5,
            selected_document_counts: {
              'document:functional': 3,
              'document:realization': 2,
            },
            primary_document_key: 'document:functional',
            primary_selected_count: 3,
            quota_relaxed: false,
          },
        },
      },
    },
  ];
}
