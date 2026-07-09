import { Component } from '@angular/core';

@Component({
  selector: 'app-shell',
  standalone: true,
  template: `
    <main class="shell">
      <header class="hero">
        <div>
          <p class="eyebrow">Phase 1 · Agent-ready baseline RAG</p>
          <h1>Indexer Core Console</h1>
          <p class="summary">
            Upload source documents, ask grounded questions, and inspect the returned evidence,
            citations, and graph execution trace.
          </p>
        </div>
        <div class="hero-card" aria-label="Current baseline pipeline">
          <span class="hero-card__label">Pipeline</span>
          <strong>retrieve → generate_answer</strong>
          <span class="hero-card__hint">Graph runner boundary remains visible for future agentic steps.</span>
        </div>
      </header>
      <ng-content />
    </main>
  `,
  styles: [
    `
      .shell {
        width: min(1440px, calc(100% - 32px));
        margin: 0 auto;
        padding: 32px 0 56px;
      }

      .hero {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 320px;
        gap: 24px;
        align-items: stretch;
        margin-bottom: 24px;
      }

      .eyebrow {
        margin: 0 0 12px;
        color: var(--primary);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      h1 {
        margin: 0;
        font-size: clamp(2.25rem, 5vw, 4.5rem);
        line-height: 0.95;
        letter-spacing: -0.06em;
      }

      .summary {
        max-width: 780px;
        margin: 18px 0 0;
        color: var(--text-muted);
        font-size: 1.05rem;
        line-height: 1.6;
      }

      .hero-card {
        display: flex;
        min-height: 180px;
        flex-direction: column;
        justify-content: center;
        gap: 12px;
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        background: var(--surface);
        box-shadow: var(--shadow);
        padding: 24px;
      }

      .hero-card__label {
        color: var(--text-muted);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .hero-card strong {
        color: var(--primary);
        font-size: 1.2rem;
      }

      .hero-card__hint {
        color: var(--text-muted);
        line-height: 1.5;
      }

      @media (max-width: 880px) {
        .shell {
          width: min(100% - 24px, 1440px);
          padding-top: 24px;
        }

        .hero {
          grid-template-columns: 1fr;
        }
      }
    `,
  ],
})
export class ShellComponent {}
