import { NgClass } from '@angular/common';
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-status-badge',
  standalone: true,
  imports: [NgClass],
  templateUrl: './status-badge.component.html',
})
export class StatusBadgeComponent {
  @Input({ required: true }) status = 'unknown';

  get label(): string {
    return this.status.replaceAll('_', ' ');
  }

  get statusClass(): string {
    const normalized = this.status.toLowerCase();
    if (['completed', 'ready', 'indexed', 'success', 'succeeded'].includes(normalized)) {
      return 'border-success/20 bg-success/10 text-success';
    }
    if (['failed', 'error'].includes(normalized)) {
      return 'border-danger/20 bg-danger/10 text-danger';
    }
    if (['running', 'processing', 'pending'].includes(normalized)) {
      return 'border-warning/20 bg-warning/10 text-warning';
    }
    return 'border-border bg-surface-muted text-text-muted';
  }
}
