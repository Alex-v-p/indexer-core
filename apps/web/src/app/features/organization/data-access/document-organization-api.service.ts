import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ContentGroup,
  ContentGroupAlias,
  DocumentContentGroupAssignment,
  DocumentOrganization,
  DocumentType,
  DocumentTypeDecision,
  ManualDocumentTypeDecision,
  OrganizationEnqueueResponse,
} from '../models/document-organization.models';

@Injectable({ providedIn: 'root' })
export class DocumentOrganizationApiService {
  private readonly http = inject(HttpClient);

  listDocumentTypes(includeArchived = false): Observable<DocumentType[]> {
    return this.http.get<DocumentType[]>('/document-types', {
      params: { include_archived: includeArchived },
    });
  }

  createDocumentType(payload: {
    key: string;
    label: string;
    description?: string;
  }): Observable<DocumentType> {
    return this.http.post<DocumentType>('/document-types', payload);
  }

  updateDocumentType(
    documentTypeId: string,
    payload: { label?: string; description?: string; archive?: boolean },
  ): Observable<DocumentType> {
    return this.http.patch<DocumentType>(`/document-types/${documentTypeId}`, payload);
  }

  listContentGroups(includeArchived = false): Observable<ContentGroup[]> {
    return this.http.get<ContentGroup[]>('/content-groups', {
      params: { include_archived: includeArchived },
    });
  }

  createContentGroup(payload: {
    name: string;
    description?: string;
  }): Observable<ContentGroup> {
    return this.http.post<ContentGroup>('/content-groups', payload);
  }

  updateContentGroup(
    contentGroupId: string,
    payload: { name?: string; archive?: boolean },
  ): Observable<ContentGroup> {
    return this.http.patch<ContentGroup>(`/content-groups/${contentGroupId}`, payload);
  }

  addContentGroupAlias(contentGroupId: string, name: string): Observable<ContentGroupAlias> {
    return this.http.post<ContentGroupAlias>(`/content-groups/${contentGroupId}/aliases`, {
      name,
    });
  }

  archiveContentGroupAlias(
    contentGroupId: string,
    aliasId: string,
  ): Observable<ContentGroupAlias> {
    return this.http.delete<ContentGroupAlias>(
      `/content-groups/${contentGroupId}/aliases/${aliasId}`,
    );
  }

  getDocumentOrganization(documentId: string): Observable<DocumentOrganization> {
    return this.http.get<DocumentOrganization>(`/documents/${documentId}/organization`);
  }

  replaceDocumentTypes(
    documentId: string,
    decisions: ManualDocumentTypeDecision[],
  ): Observable<{ decisions: DocumentTypeDecision[] }> {
    return this.http.put<{ decisions: DocumentTypeDecision[] }>(
      `/documents/${documentId}/organization/types`,
      { decisions },
    );
  }

  setDocumentContentGroup(
    documentId: string,
    contentGroupId: string | null,
    expectedRevision: number,
  ): Observable<DocumentContentGroupAssignment | null> {
    return this.http.put<DocumentContentGroupAssignment | null>(
      `/documents/${documentId}/organization/content-group`,
      {
        content_group_id: contentGroupId,
        expected_revision: expectedRevision,
      },
    );
  }

  requeueDocument(documentId: string): Observable<OrganizationEnqueueResponse> {
    return this.http.post<OrganizationEnqueueResponse>(
      `/documents/${documentId}/organization/requeue`,
      {},
    );
  }

  backfill(limit: number): Observable<OrganizationEnqueueResponse> {
    return this.http.post<OrganizationEnqueueResponse>(
      '/document-organization/backfill',
      {},
      { params: { limit } },
    );
  }
}
