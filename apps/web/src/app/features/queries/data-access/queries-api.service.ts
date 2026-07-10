import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { QueryRequest, QueryResponse } from '../models/query.models';

@Injectable({ providedIn: 'root' })
export class QueriesApiService {
  private readonly http = inject(HttpClient);

  createQueryRun(payload: QueryRequest): Observable<QueryResponse> {
    return this.http.post<QueryResponse>('/queries', payload);
  }

  getQueryRun(queryRunId: string): Observable<QueryResponse> {
    return this.http.get<QueryResponse>(`/queries/${queryRunId}`);
  }
}
