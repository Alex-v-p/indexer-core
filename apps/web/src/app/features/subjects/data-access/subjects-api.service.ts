import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  DocumentSubjectDecision,
  Subject,
  SubjectAlias,
  SubjectDecisionState,
  SubjectKind,
} from '../models/subject.models';

@Injectable({ providedIn: 'root' })
export class SubjectsApiService {
  private readonly http = inject(HttpClient);

  listSubjects(includeArchived = false): Observable<Subject[]> {
    return this.http.get<Subject[]>('/subjects', {
      params: { include_archived: includeArchived },
    });
  }

  createSubject(payload: {
    kind: SubjectKind;
    name: string;
    description?: string;
  }): Observable<Subject> {
    return this.http.post<Subject>('/subjects', payload);
  }

  updateSubject(
    subjectId: string,
    payload: { name?: string; archive?: boolean },
  ): Observable<Subject> {
    return this.http.patch<Subject>(`/subjects/${subjectId}`, payload);
  }

  addAlias(subjectId: string, name: string): Observable<SubjectAlias> {
    return this.http.post<SubjectAlias>(`/subjects/${subjectId}/aliases`, { name });
  }

  archiveAlias(subjectId: string, aliasId: string): Observable<SubjectAlias> {
    return this.http.delete<SubjectAlias>(`/subjects/${subjectId}/aliases/${aliasId}`);
  }

  listDocumentDecisions(documentId: string): Observable<DocumentSubjectDecision[]> {
    return this.http.get<DocumentSubjectDecision[]>(`/documents/${documentId}/subjects`);
  }

  listDocumentSuggestions(documentId: string): Observable<DocumentSubjectDecision[]> {
    return this.http.get<DocumentSubjectDecision[]>(
      `/documents/${documentId}/subject-suggestions`,
    );
  }

  setDocumentDecision(
    documentId: string,
    subjectId: string,
    state: Extract<SubjectDecisionState, 'assigned' | 'rejected'>,
    expectedRevision: number,
    rationale?: string,
  ): Observable<DocumentSubjectDecision> {
    return this.http.put<DocumentSubjectDecision>(
      `/documents/${documentId}/subjects/${subjectId}`,
      {
        state,
        expected_revision: expectedRevision,
        rationale: rationale || undefined,
      },
    );
  }

  reviewSuggestion(
    documentId: string,
    subjectId: string,
    decision: 'accept' | 'reject',
    expectedRevision: number,
  ): Observable<DocumentSubjectDecision> {
    return this.http.post<DocumentSubjectDecision>(
      `/documents/${documentId}/subject-suggestions/${subjectId}/review`,
      { decision, expected_revision: expectedRevision },
    );
  }
}
