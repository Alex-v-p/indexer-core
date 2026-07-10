import { NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { QueryRequest } from '../../models/query.models';

@Component({
  selector: 'app-query-input',
  standalone: true,
  imports: [FormsModule, NgIf],
  templateUrl: './query-input.component.html',
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
