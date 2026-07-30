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
      },
    ]);
    expect(emitted[0]).not.toHaveProperty('top_k');
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
