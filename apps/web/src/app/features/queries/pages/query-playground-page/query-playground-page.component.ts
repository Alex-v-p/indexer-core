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
  template: `
    <section class="page-stack" aria-label="Question workspace">
      <header class="page-heading">
        <div>
          <p class="page-heading__eyebrow">Ask</p>
          <h1>Questions</h1>
          <p>
            Ask questions against the indexed document library and inspect the answer,
            citations, supporting evidence, and execution trace returned by the query workflow.
          </p>
        </div>
        <span class="pipeline-pill">retrieve → generate answer</span>
      </header>

      <section class="panel query-panel">
        <div class="panel__header">
          <div>
            <p class="panel__eyebrow">Query</p>
            <h2>Ask a grounded question</h2>
          </div>
        </div>

        <app-query-input
          [running]="queryRunning()"
          [error]="queryError()"
          (questionAsked)="askQuestion($event)"
        />

        <app-answer-panel *ngIf="queryResult() as result" [result]="result" />

        <p class="empty-state" *ngIf="!queryResult() && !queryRunning()">
          Answers, citations, evidence, and trace steps will appear here after a query run.
        </p>
      </section>
    </section>
  `,
  styles: [
    `
      .page-stack {
        display: grid;
        gap: 24px;
      }

      .page-heading {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 24px;
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        background: var(--surface);
        box-shadow: var(--shadow);
        padding: 28px;
      }

      .page-heading__eyebrow,
      .panel__eyebrow {
        margin: 0 0 6px;
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      h1,
      h2 {
        margin: 0;
        letter-spacing: -0.04em;
      }

      h1 {
        font-size: clamp(2.25rem, 5vw, 4.25rem);
        line-height: 0.95;
      }

      h2 {
        font-size: 1.35rem;
      }

      .page-heading p:not(.page-heading__eyebrow) {
        max-width: 820px;
        margin: 16px 0 0;
        color: var(--text-muted);
        line-height: 1.6;
      }

      .panel {
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        background: var(--surface);
        box-shadow: var(--shadow);
        padding: 24px;
      }

      .query-panel {
        max-width: 980px;
      }

      .panel__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .pipeline-pill {
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary);
        padding: 8px 14px;
        font-size: 0.85rem;
        font-weight: 800;
        white-space: nowrap;
      }

      .empty-state {
        margin: 20px 0 0;
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius-md);
        color: var(--text-muted);
        padding: 18px;
        line-height: 1.5;
      }

      @media (max-width: 760px) {
        .page-heading {
          align-items: stretch;
          flex-direction: column;
          padding: 22px;
        }

        .panel {
          padding: 18px;
        }
      }
    `,
  ],
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
