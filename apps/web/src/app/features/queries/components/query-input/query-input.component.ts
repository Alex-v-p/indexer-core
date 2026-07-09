import { NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { QueryRequest } from '../../models/query.models';

@Component({
  selector: 'app-query-input',
  standalone: true,
  imports: [FormsModule, NgIf],
  template: `
    <form class="query-form" (ngSubmit)="submitQuestion()">
      <label>
        <span>Question</span>
        <textarea
          name="question"
          [(ngModel)]="question"
          rows="5"
          placeholder="Ask something that should be answered from the uploaded documents…"
          required
        ></textarea>
      </label>

      <div class="query-form__controls">
        <label class="top-k-control">
          <span>Top-k chunks</span>
          <input type="number" name="topK" [(ngModel)]="topK" min="1" max="25" />
        </label>
        <button class="primary-button" type="submit" [disabled]="question.trim().length === 0 || running">
          {{ running ? 'Running graph…' : 'Ask question' }}
        </button>
      </div>

      <p class="error-message" *ngIf="error">{{ error }}</p>
    </form>
  `,
  styles: [
    `
      .query-form {
        display: grid;
        gap: 16px;
        margin-top: 20px;
      }

      label {
        display: grid;
        gap: 8px;
        color: var(--text-muted);
        font-size: 0.86rem;
        font-weight: 700;
      }

      input,
      textarea {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        background: var(--surface);
        color: var(--text);
        padding: 12px 14px;
        outline: none;
        transition: border-color 160ms ease, box-shadow 160ms ease;
      }

      textarea {
        resize: vertical;
      }

      input:focus,
      textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px var(--primary-soft);
      }

      .query-form__controls {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 16px;
      }

      .top-k-control {
        max-width: 160px;
      }

      .primary-button {
        display: inline-flex;
        min-height: 44px;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: var(--primary);
        color: white;
        padding: 0 20px;
        font-weight: 800;
        transition: background 160ms ease, transform 160ms ease;
      }

      .primary-button:not(:disabled):hover {
        background: var(--primary-strong);
        transform: translateY(-1px);
      }

      .error-message {
        margin: 0;
        color: var(--danger);
        font-weight: 700;
      }

      @media (max-width: 640px) {
        .query-form__controls {
          align-items: stretch;
          flex-direction: column;
        }

        .top-k-control {
          max-width: none;
        }
      }
    `,
  ],
})
export class QueryInputComponent {
  @Input() running = false;
  @Input() error: string | null = null;
  @Output() questionAsked = new EventEmitter<QueryRequest>();

  question = '';
  topK = 5;

  submitQuestion(): void {
    const normalizedQuestion = this.question.trim();
    if (!normalizedQuestion) {
      return;
    }

    const safeTopK = Math.min(Math.max(Number(this.topK) || 5, 1), 25);
    this.topK = safeTopK;
    this.questionAsked.emit({ question: normalizedQuestion, top_k: safeTopK });
  }
}
