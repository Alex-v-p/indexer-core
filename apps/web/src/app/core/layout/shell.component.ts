import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [RouterLink, RouterLinkActive],
  template: `
    <header class="app-header">
      <div class="app-header__inner">
        <a class="brand" routerLink="/questions" aria-label="Indexer Core home">
          <span class="brand__mark">IC</span>
          <span>
            <strong>Indexer Core</strong>
            <small>Document intelligence workspace</small>
          </span>
        </a>

        <nav class="nav" aria-label="Primary navigation">
          <a routerLink="/documents" routerLinkActive="nav__link--active">Documents</a>
          <a routerLink="/questions" routerLinkActive="nav__link--active">Questions</a>
        </nav>
      </div>
    </header>

    <main class="shell">
      <ng-content />
    </main>
  `,
  styles: [
    `
      .app-header {
        position: sticky;
        top: 0;
        z-index: 20;
        border-bottom: 1px solid var(--border);
        background: rgb(245 247 251 / 88%);
        backdrop-filter: blur(18px);
      }

      .app-header__inner {
        display: flex;
        width: min(1440px, calc(100% - 32px));
        min-height: 76px;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        margin: 0 auto;
      }

      .brand {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        color: inherit;
        text-decoration: none;
      }

      .brand__mark {
        display: grid;
        width: 42px;
        height: 42px;
        place-items: center;
        border-radius: 14px;
        background: var(--primary);
        color: white;
        font-weight: 900;
        letter-spacing: -0.08em;
      }

      .brand strong,
      .brand small {
        display: block;
      }

      .brand strong {
        font-size: 1rem;
        letter-spacing: -0.03em;
      }

      .brand small {
        margin-top: 2px;
        color: var(--text-muted);
        font-size: 0.78rem;
      }

      .nav {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--surface);
        padding: 6px;
        box-shadow: 0 10px 28px rgb(23 32 51 / 6%);
      }

      .nav a {
        border-radius: 999px;
        color: var(--text-muted);
        padding: 10px 16px;
        font-size: 0.9rem;
        font-weight: 800;
        text-decoration: none;
        transition: background 160ms ease, color 160ms ease;
      }

      .nav a:hover,
      .nav__link--active {
        background: var(--primary-soft);
        color: var(--primary) !important;
      }

      .shell {
        width: min(1440px, calc(100% - 32px));
        margin: 0 auto;
        padding: 32px 0 56px;
      }

      @media (max-width: 640px) {
        .app-header__inner {
          width: min(100% - 24px, 1440px);
          min-height: auto;
          align-items: stretch;
          flex-direction: column;
          padding: 14px 0;
        }

        .nav {
          width: 100%;
        }

        .nav a {
          flex: 1;
          text-align: center;
        }

        .shell {
          width: min(100% - 24px, 1440px);
          padding-top: 24px;
        }
      }
    `,
  ],
})
export class ShellComponent {}
