import { NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { PipelineSummary, QueryRequest } from '../../models/query.models';
import { Subject } from '../../../subjects/models/subject.models';

@Component({
  selector: 'app-query-input',
  standalone: true,
  imports: [FormsModule, NgFor, NgIf],
  templateUrl: './query-input.component.html',
})
export class QueryInputComponent implements OnChanges {
  @Input() running = false;
  @Input() error: string | null = null;
  @Input() pipelines: PipelineSummary[] = [];
  @Input() defaultPipelineName: string | null = null;
  @Input() pipelinesLoading = false;
  @Input() pipelinesError: string | null = null;
  @Input() subjects: Subject[] = [];
  @Input() subjectsLoading = false;
  @Input() subjectsError: string | null = null;
  @Output() questionAsked = new EventEmitter<QueryRequest>();

  question = '';
  pipelineName = '';
  selectedSubjectIds: string[] = [];
  coverageMode: 'best_evidence' | 'multi_document' = 'best_evidence';

  ngOnChanges(changes: SimpleChanges): void {
    if (!changes['pipelines'] && !changes['defaultPipelineName']) {
      return;
    }

    const selectedExists = this.pipelines.some((pipeline) => pipeline.name === this.pipelineName);
    if (!selectedExists) {
      this.pipelineName = this.defaultPipelineName ?? this.pipelines[0]?.name ?? '';
    }
  }

  submitQuestion(): void {
    const normalizedQuestion = this.question.trim();
    if (!normalizedQuestion) {
      return;
    }

    this.questionAsked.emit({
      question: normalizedQuestion,
      pipeline_name: this.pipelineName || null,
      subject_ids: [...this.selectedSubjectIds],
      coverage_mode: this.coverageMode,
    });
  }

  isSubjectSelected(subjectId: string): boolean {
    return this.selectedSubjectIds.includes(subjectId);
  }

  toggleSubject(subjectId: string, selected: boolean): void {
    this.selectedSubjectIds = selected
      ? [...new Set([...this.selectedSubjectIds, subjectId])]
      : this.selectedSubjectIds.filter((id) => id !== subjectId);
  }

  pipelineTrackBy(index: number, pipeline: PipelineSummary): string {
    return pipeline.name || String(index);
  }

  subjectTrackBy(index: number, subject: Subject): string {
    return subject.id || String(index);
  }
}
