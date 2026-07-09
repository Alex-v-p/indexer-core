import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { StatusBadgeComponent } from '../../../../shared/ui/status-badge.component';
import { shortId } from '../../../../shared/utils/formatting';
import { CitationListComponent } from '../citation-list/citation-list.component';
import { EvidenceViewerComponent } from '../evidence-viewer/evidence-viewer.component';
import { QueryResponse, TraceStep } from '../../models/query.models';

@Component({
  selector: 'app-answer-panel',
  standalone: true,
  imports: [NgFor, NgIf, StatusBadgeComponent, CitationListComponent, EvidenceViewerComponent],
  template: `
    <article class="result-card" *ngIf="result as currentResult">
      <header class="result-card__header">
        <div>
          <p class="panel__eyebrow">Answer</p>
          <h3>{{ currentResult.question }}</h3>
        </div>
        <app-status-badge [status]="currentResult.status" />
      </header>

      <div class="answer-box">
        <p>{{ currentResult.answer || 'No answer was returned.' }}</p>
      </div>

      <div class="result-meta">
        <span>Run {{ shortId(currentResult.id) }}</span>
        <span>{{ currentResult.pipeline_name || 'pipeline' }} {{ currentResult.pipeline_version || '' }}</span>
        <span>{{ currentResult.evidence.length }} evidence items</span>
        <span>{{ currentResult.trace.length }} trace steps</span>
      </div>

      <app-citation-list [citations]="currentResult.citations" />
      <app-evidence-viewer [evidence]="currentResult.evidence" />

      <section class="result-section" *ngIf="currentResult.trace.length > 0">
        <div class="section-heading">
          <h4>Execution trace</h4>
          <span>{{ currentResult.trace.length }}</span>
        </div>
        <ol class="trace-list">
          <li *ngFor="let step of currentResult.trace; trackBy: trackTrace">
            <div class="trace-step__marker">{{ step.step_order }}</div>
            <div class="trace-step__body">
              <div class="trace-step__topline">
                <strong>{{ step.name }}</strong>
                <app-status-badge [status]="step.status" />
              </div>
              <p>
                <span>{{ step.step_type || 'step' }}</span>
                <span *ngIf="step.duration_ms !== null"> · {{ step.duration_ms }} ms</span>
              </p>
              <small *ngIf="step.input_summary">Input: {{ step.input_summary }}</small>
              <small *ngIf="step.output_summary">Output: {{ step.output_summary }}</small>
              <small class="error-message" *ngIf="step.error_message">{{ step.error_message }}</small>
            </div>
          </li>
        </ol>
      </section>
    </article>
  `,
  styles: [
    `
      .result-card {
        display: grid;
        gap: 16px;
        margin-top: 20px;
      }

      .result-card__header,
      .section-heading,
      .trace-step__topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .panel__eyebrow {
        margin: 0 0 6px;
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      h3,
      h4 {
        margin: 0;
        letter-spacing: -0.03em;
      }

      h3 {
        font-size: 1.05rem;
      }

      h4 {
        font-size: 1rem;
      }

      .answer-box,
      .trace-list li {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--surface-muted);
        padding: 16px;
      }

      .answer-box {
        background: linear-gradient(180deg, #f8faff 0%, #ffffff 100%);
      }

      .answer-box p {
        margin: 0;
        font-size: 1rem;
        line-height: 1.7;
        white-space: pre-wrap;
      }

      .result-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        color: var(--text-muted);
        font-size: 0.83rem;
      }

      .result-meta span {
        border-radius: 999px;
        background: var(--surface-muted);
        padding: 6px 10px;
      }

      .result-section {
        display: grid;
        gap: 12px;
        margin-top: 6px;
      }

      .section-heading span {
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary);
        padding: 6px 12px;
        font-size: 0.8rem;
        font-weight: 800;
      }

      .trace-list {
        display: grid;
        gap: 10px;
        margin: 12px 0 0;
        padding: 0;
        list-style: none;
      }

      .trace-list li {
        display: grid;
        grid-template-columns: 36px minmax(0, 1fr);
        gap: 12px;
      }

      .trace-step__marker {
        display: grid;
        width: 36px;
        height: 36px;
        place-items: center;
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary);
        font-weight: 900;
      }

      .trace-step__body {
        display: grid;
        min-width: 0;
        gap: 6px;
      }

      .trace-step__body p {
        margin: 0;
        color: var(--text-muted);
        font-size: 0.83rem;
      }

      .trace-step__body small {
        color: var(--text-muted);
        font-size: 0.83rem;
      }

      .error-message {
        margin: 0;
        color: var(--danger);
        font-weight: 700;
      }

      @media (max-width: 640px) {
        .result-card__header {
          align-items: stretch;
          flex-direction: column;
        }
      }
    `,
  ],
})
export class AnswerPanelComponent {
  @Input() result: QueryResponse | null = null;

  readonly shortId = shortId;

  trackTrace(index: number, step: TraceStep): string {
    return step.id ?? `${step.step_order}-${step.name}-${index}`;
  }
}
