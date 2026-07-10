import { NgIf } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { finalize } from 'rxjs';

import { toApiErrorMessage } from '../../../../core/http/api-error';
import { AnswerPanelComponent } from '../../components/answer-panel/answer-panel.component';
import { QueryInputComponent } from '../../components/query-input/query-input.component';
import { QueriesApiService } from '../../data-access/queries-api.service';
import { PipelineSummary, QueryRequest, QueryResponse } from '../../models/query.models';

@Component({
  selector: 'app-query-playground-page',
  standalone: true,
  imports: [NgIf, QueryInputComponent, AnswerPanelComponent],
  templateUrl: './query-playground-page.component.html',
})
export class QueryPlaygroundPageComponent implements OnInit {
  private readonly queriesApi = inject(QueriesApiService);

  readonly queryRunning = signal(false);
  readonly queryError = signal<string | null>(null);
  readonly queryResult = signal<QueryResponse | null>(null);
  readonly pipelines = signal<PipelineSummary[]>([]);
  readonly defaultPipelineName = signal<string | null>(null);
  readonly pipelinesLoading = signal(false);
  readonly pipelinesError = signal<string | null>(null);

  ngOnInit(): void {
    this.loadPipelines();
  }

  askQuestion(request: QueryRequest): void {
    this.queryRunning.set(true);
    this.queryError.set(null);

    this.queriesApi
      .createQueryRun(request)
      .pipe(finalize(() => this.queryRunning.set(false)))
      .subscribe({
        next: (result) => this.queryResult.set(result),
        error: (error: unknown) => this.queryError.set(toApiErrorMessage(error)),
      });
  }

  private loadPipelines(): void {
    this.pipelinesLoading.set(true);
    this.pipelinesError.set(null);

    this.queriesApi
      .listPipelines()
      .pipe(finalize(() => this.pipelinesLoading.set(false)))
      .subscribe({
        next: (result) => {
          this.pipelines.set(result.pipelines);
          this.defaultPipelineName.set(result.default_pipeline_name);
        },
        error: (error: unknown) => {
          this.pipelinesError.set(toApiErrorMessage(error));
        },
      });
  }
}
