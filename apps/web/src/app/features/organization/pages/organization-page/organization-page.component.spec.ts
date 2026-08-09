import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { DocumentOrganizationApiService } from '../../data-access/document-organization-api.service';
import { OrganizationPageComponent } from './organization-page.component';

describe('OrganizationPageComponent', () => {
  it('loads content groups and extensible document types', async () => {
    await TestBed.configureTestingModule({
      imports: [OrganizationPageComponent],
      providers: [{
        provide: DocumentOrganizationApiService,
        useValue: {
          listContentGroups: () => of([makeGroup()]),
          listDocumentTypes: () => of([makeType()]),
        },
      }],
    }).compileComponents();
    const fixture = TestBed.createComponent(OrganizationPageComponent);
    fixture.detectChanges();

    expect(fixture.componentInstance.contentGroups()).toHaveLength(1);
    expect(fixture.componentInstance.documentTypes()).toHaveLength(1);
    expect(fixture.nativeElement.querySelector('h1').textContent).toContain(
      'Content groups & types',
    );
  });

  it('manages aliases on the selected content group', async () => {
    const addAlias = jest.fn(() => of({
      id: 'alias-1', content_group_id: 'group-1', name: 'Project DAF',
      normalized_name: 'project daf', archived_at: null, created_at: '',
    }));
    await TestBed.configureTestingModule({
      imports: [OrganizationPageComponent],
      providers: [{
        provide: DocumentOrganizationApiService,
        useValue: {
          listContentGroups: () => of([]),
          listDocumentTypes: () => of([]),
          addContentGroupAlias: addAlias,
        },
      }],
    }).compileComponents();
    const component = TestBed.createComponent(OrganizationPageComponent).componentInstance;
    component.contentGroups.set([makeGroup()]);
    component.selectGroup(makeGroup());
    component.aliasName = 'Project DAF';

    component.addAlias();

    expect(addAlias).toHaveBeenCalledWith('group-1', 'Project DAF');
    expect(component.selectedGroup()?.aliases[0]?.name).toBe('Project DAF');
  });

  it('bounds the automatic-classification backfill', async () => {
    const backfill = jest.fn(() => of({ job_ids: [], skipped_document_ids: ['document-1'] }));
    await TestBed.configureTestingModule({
      imports: [OrganizationPageComponent],
      providers: [{
        provide: DocumentOrganizationApiService,
        useValue: {
          listContentGroups: () => of([]),
          listDocumentTypes: () => of([]),
          backfill,
        },
      }],
    }).compileComponents();
    const component = TestBed.createComponent(OrganizationPageComponent).componentInstance;
    component.backfillLimit = 9000;

    component.startBackfill();

    expect(backfill).toHaveBeenCalledWith(5000);
    expect(component.message()).toContain('No jobs were needed');
  });
});

function makeGroup() {
  return {
    id: 'group-1', name: 'DAF', normalized_name: 'daf', description: null,
    metadata: {}, archived_at: null, created_at: '', updated_at: '', aliases: [],
  };
}

function makeType() {
  return {
    id: 'type-1', key: 'report', label: 'Report', description: null,
    metadata: {}, archived_at: null, created_at: '', updated_at: '',
  };
}
