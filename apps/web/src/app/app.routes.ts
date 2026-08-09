import { Routes } from '@angular/router';

import { DocumentsPageComponent } from './features/documents/pages/documents-page/documents-page.component';
import { QueryPlaygroundPageComponent } from './features/queries/pages/query-playground-page/query-playground-page.component';
import { OrganizationPageComponent } from './features/organization/pages/organization-page/organization-page.component';

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
    path: 'organization',
    component: OrganizationPageComponent,
    title: 'Content groups & types · Indexer Core',
  },
  {
    path: 'subjects',
    redirectTo: 'organization',
  },
  {
    path: '**',
    redirectTo: 'questions',
  },
];
