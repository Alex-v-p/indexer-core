import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { ShellComponent } from './core/layout/shell.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, ShellComponent],
  templateUrl: './app.component.html',
})
export class AppComponent {}
