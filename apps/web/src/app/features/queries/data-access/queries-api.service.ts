import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { PipelineListResponse, QueryRequest, QueryResponse, QueuedQueryResponse } from '../models/query.models';

@Injectable({ providedIn: 'root' })
export class QueriesApiService {
  private readonly http = inject(HttpClient);

  listPipelines(): Observable<PipelineListResponse> {
    return this.http.get<PipelineListResponse>('/pipelines');
  }

  createQueryRun(payload: QueryRequest): Observable<QueuedQueryResponse> {
    return this.http.post<QueuedQueryResponse>('/queries', payload);
  }

  getQueryRun(queryRunId: string): Observable<QueryResponse> {
    return this.http.get<QueryResponse>(`/queries/${queryRunId}`);
  }
}
