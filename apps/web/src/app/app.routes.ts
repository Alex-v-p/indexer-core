import { Routes } from '@angular/router';

import { DocumentsPageComponent } from './features/documents/pages/documents-page/documents-page.component';
import { QueryPlaygroundPageComponent } from './features/queries/pages/query-playground-page/query-playground-page.component';
import { SubjectsPageComponent } from './features/subjects/pages/subjects-page/subjects-page.component';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: 'questions',
  },
  {
    path: 'documents',
    component: DocumentsPageComponent,
    title: 'Documents · Indexer Core',
  },
  {
    path: 'questions',
    component: QueryPlaygroundPageComponent,
    title: 'Questions · Indexer Core',
  },
  {
    path: 'subjects',
    component: SubjectsPageComponent,
    title: 'Subjects · Indexer Core',
  },
  {
    path: '**',
    redirectTo: 'questions',
  },
];
