import { Routes } from '@angular/router';

import { QueryPlaygroundPageComponent } from './features/queries/pages/query-playground-page/query-playground-page.component';

export const routes: Routes = [
  {
    path: '',
    component: QueryPlaygroundPageComponent,
    title: 'Indexer Core Console',
  },
  {
    path: '**',
    redirectTo: '',
  },
];
