import { NgClass } from '@angular/common';
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-status-badge',
  standalone: true,
  imports: [NgClass],
  template: `
    <span class="status-badge" [ngClass]="statusClass">
      {{ label }}
    </span>
  `,
  styles: [
    `
      .status-badge {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--surface-muted);
        color: var(--text-muted);
        padding: 4px 10px;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: capitalize;
      }

      .status-badge--success {
        border-color: rgb(20 125 82 / 18%);
        background: rgb(20 125 82 / 10%);
        color: var(--success);
      }

      .status-badge--warning {
        border-color: rgb(161 92 0 / 20%);
        background: rgb(161 92 0 / 10%);
        color: var(--warning);
      }

      .status-badge--danger {
        border-color: rgb(180 35 24 / 18%);
        background: rgb(180 35 24 / 10%);
        color: var(--danger);
      }
    `,
  ],
})
export class StatusBadgeComponent {
  @Input({ required: true }) status = 'unknown';

  get label(): string {
    return this.status.replaceAll('_', ' ');
  }

  get statusClass(): string {
    const normalized = this.status.toLowerCase();
    if (['completed', 'ready', 'indexed', 'success', 'succeeded'].includes(normalized)) {
      return 'status-badge--success';
    }
    if (['failed', 'error'].includes(normalized)) {
      return 'status-badge--danger';
    }
    if (['running', 'processing', 'pending'].includes(normalized)) {
      return 'status-badge--warning';
    }
    return '';
  }
}
