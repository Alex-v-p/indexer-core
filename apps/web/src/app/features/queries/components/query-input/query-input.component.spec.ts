import { QueryRequest } from '../../models/query.models';
import { QueryInputComponent } from './query-input.component';

describe('QueryInputComponent', () => {
  it('emits a trimmed web request without top-k', () => {
    const component = new QueryInputComponent();
    const emitted: QueryRequest[] = [];
    component.question = '  What is indexed?  ';
    component.pipelineName = 'agentic_rag';
    component.questionAsked.subscribe((request) => emitted.push(request));

    component.submitQuestion();

    expect(emitted).toEqual([
      {
        question: 'What is indexed?',
        pipeline_name: 'agentic_rag',
        subject_ids: [],
        coverage_mode: 'best_evidence',
      },
    ]);
    expect(emitted[0]).not.toHaveProperty('top_k');
  });

  it('submits explicit subjects and multi-document coverage additively', () => {
    const component = new QueryInputComponent();
    const emitted: QueryRequest[] = [];
    component.question = 'Compare delivery';
    component.toggleSubject('project-1', true);
    component.toggleSubject('topic-1', true);
    component.coverageMode = 'multi_document';
    component.questionAsked.subscribe((request) => emitted.push(request));

    component.submitQuestion();

    expect(emitted[0]).toMatchObject({
      subject_ids: ['project-1', 'topic-1'],
      coverage_mode: 'multi_document',
    });
  });

  it('does not emit a request for a blank question', () => {
    const component = new QueryInputComponent();
    const emitted: QueryRequest[] = [];
    component.question = '   ';
    component.questionAsked.subscribe((request) => emitted.push(request));

    component.submitQuestion();

    expect(emitted).toEqual([]);
  });
});
