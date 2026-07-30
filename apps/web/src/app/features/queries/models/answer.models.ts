export interface AnswerPresentation {
  schema_version: '1.0';
  outcome:
    | 'complete'
    | 'partial'
    | 'blocked_constraint_no_match'
    | 'blocked_insufficient_evidence'
    | 'blocked_no_evidence';
  title: string;
  body: string;
  supported_information: string[];
  unresolved_information: string[];
  citation_count: number;
}
