import { NgIf } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import { finalize } from 'rxjs';

import { toApiErrorMessage } from '../../../../core/http/api-error';
import { AnswerPanelComponent } from '../../components/answer-panel/answer-panel.component';
import { QueryInputComponent } from '../../components/query-input/query-input.component';
import { QueriesApiService } from '../../data-access/queries-api.service';
import { QueryRequest, QueryResponse } from '../../models/query.models';

@Component({
  selector: 'app-query-playground-page',
  standalone: true,
  imports: [NgIf, QueryInputComponent, AnswerPanelComponent],
  templateUrl: './query-playground-page.component.html',
})
export class QueryPlaygroundPageComponent {
  private readonly queriesApi = inject(QueriesApiService);

  readonly queryRunning = signal(false);
  readonly queryError = signal<string | null>(null);
  readonly queryResult = signal<QueryResponse | null>(null);

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
}
