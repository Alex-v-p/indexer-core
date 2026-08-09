import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { DocumentOrganizationApiService } from './document-organization-api.service';

describe('DocumentOrganizationApiService', () => {
  let service: DocumentOrganizationApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        DocumentOrganizationApiService,
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    service = TestBed.inject(DocumentOrganizationApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('sends explicit revisions for manual group and multi-type changes', () => {
    service.setDocumentContentGroup('document-1', 'group-1', 4).subscribe();
    const group = http.expectOne('/documents/document-1/organization/content-group');
    expect(group.request.method).toBe('PUT');
    expect(group.request.body).toEqual({
      content_group_id: 'group-1',
      expected_revision: 4,
    });
    group.flush({});

    service.replaceDocumentTypes('document-1', [
      { document_type_id: 'type-1', state: 'assigned', expected_revision: 2 },
      { document_type_id: 'type-2', state: 'rejected', expected_revision: 1 },
    ]).subscribe();
    const types = http.expectOne('/documents/document-1/organization/types');
    expect(types.request.method).toBe('PUT');
    expect(types.request.body.decisions).toHaveLength(2);
    types.flush({ decisions: [] });
  });

  it('uses the additive catalogue and reclassification endpoints', () => {
    service.listContentGroups().subscribe();
    http.expectOne((request) =>
      request.url === '/content-groups' && request.params.get('include_archived') === 'false'
    ).flush([]);

    service.listDocumentTypes().subscribe();
    http.expectOne((request) =>
      request.url === '/document-types' && request.params.get('include_archived') === 'false'
    ).flush([]);

    service.requeueDocument('document-1').subscribe();
    http.expectOne('/documents/document-1/organization/requeue').flush({
      job_ids: [],
      skipped_document_ids: [],
    });
  });

  it('uses catalogue mutation endpoints for groups, aliases, and types', () => {
    service.createContentGroup({ name: 'DAF' }).subscribe();
    expect(http.expectOne('/content-groups').request.method).toBe('POST');

    service.updateContentGroup('group-1', { name: 'DAF project' }).subscribe();
    expect(http.expectOne('/content-groups/group-1').request.method).toBe('PATCH');

    service.addContentGroupAlias('group-1', 'Project DAF').subscribe();
    expect(http.expectOne('/content-groups/group-1/aliases').request.method).toBe('POST');

    service.archiveContentGroupAlias('group-1', 'alias-1').subscribe();
    expect(
      http.expectOne('/content-groups/group-1/aliases/alias-1').request.method,
    ).toBe('DELETE');

    service.createDocumentType({ key: 'report', label: 'Report' }).subscribe();
    expect(http.expectOne('/document-types').request.method).toBe('POST');

    service.updateDocumentType('type-1', { archive: true }).subscribe();
    expect(http.expectOne('/document-types/type-1').request.method).toBe('PATCH');
  });
});
