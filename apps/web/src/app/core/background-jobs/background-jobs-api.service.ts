import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { BackgroundJob } from './background-job.models';

@Injectable({ providedIn: 'root' })
export class BackgroundJobsApiService {
  private readonly http = inject(HttpClient);

  listJobs(limit = 50): Observable<BackgroundJob[]> {
    return this.http.get<BackgroundJob[]>('/jobs', { params: { limit } });
  }

  getJob(jobId: string): Observable<BackgroundJob> {
    return this.http.get<BackgroundJob>(`/jobs/${jobId}`);
  }
}
