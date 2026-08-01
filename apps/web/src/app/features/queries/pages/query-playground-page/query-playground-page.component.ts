import { NgIf } from '@angular/common';
import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { Subscription, finalize, retry, switchMap, takeWhile, timer } from 'rxjs';

import { BackgroundJob } from '../../../../core/background-jobs/background-job.models';
import { BackgroundJobsApiService } from '../../../../core/background-jobs/background-jobs-api.service';
import { toApiErrorMessage } from '../../../../core/http/api-error';
import { AnswerPanelComponent } from '../../components/answer-panel/answer-panel.component';
import { QueryInputComponent } from '../../components/query-input/query-input.component';
import { QueryJobProgressComponent } from '../../components/query-job-progress/query-job-progress.component';
import { QueriesApiService } from '../../data-access/queries-api.service';
import { PipelineSummary, QueryRequest, QueryResponse } from '../../models/query.models';
import { SubjectsApiService } from '../../../subjects/data-access/subjects-api.service';
import { Subject } from '../../../subjects/models/subject.models';

@Component({
  selector: 'app-query-playground-page',
  standalone: true,
  imports: [NgIf, QueryInputComponent, QueryJobProgressComponent, AnswerPanelComponent],
  templateUrl: './query-playground-page.component.html',
})
export class QueryPlaygroundPageComponent implements OnInit, OnDestroy {
  private readonly queriesApi = inject(QueriesApiService);
  private readonly jobsApi = inject(BackgroundJobsApiService);
  private readonly subjectsApi = inject(SubjectsApiService);
  private jobPolling: Subscription | null = null;
  private resultLoading = false;
  private subjectsRequest = 0;

  readonly queryRunning = signal(false);
  readonly queryError = signal<string | null>(null);
  readonly queryResult = signal<QueryResponse | null>(null);
  readonly queryJob = signal<BackgroundJob | null>(null);
  readonly pipelines = signal<PipelineSummary[]>([]);
  readonly defaultPipelineName = signal<string | null>(null);
  readonly pipelinesLoading = signal(false);
  readonly pipelinesError = signal<string | null>(null);
  readonly subjects = signal<Subject[]>([]);
  readonly subjectsLoading = signal(false);
  readonly subjectsError = signal<string | null>(null);

  ngOnInit(): void {
    this.loadPipelines();
    this.loadSubjects();
    this.resumeActiveQuery();
  }

  ngOnDestroy(): void {
    this.jobPolling?.unsubscribe();
    ++this.subjectsRequest;
  }

  askQuestion(request: QueryRequest): void {
    this.stopTracking();
    this.queryRunning.set(true);
    this.queryError.set(null);
    this.queryResult.set(null);
    this.queryJob.set(null);

    this.queriesApi.createQueryRun(request).subscribe({
      next: (queued) => {
        this.queryJob.set(queued.job);
        storeActiveQuery(queued.job.id, queued.query.id);
        this.trackQueryJob(queued.job.id, queued.query.id);
      },
      error: (error: unknown) => {
        this.queryRunning.set(false);
        this.queryError.set(toApiErrorMessage(error));
      },
    });
  }

  private trackQueryJob(jobId: string, queryRunId: string): void {
    this.jobPolling = timer(0, 1_000)
      .pipe(
        switchMap(() => this.jobsApi.getJob(jobId)),
        retry({ count: 5, delay: 2_000 }),
        takeWhile((job) => !isTerminal(job), true),
      )
      .subscribe({
        next: (job) => {
          this.queryJob.set(job);
          if (job.status === 'succeeded') {
            this.loadQueryResult(queryRunId);
          } else if (job.status === 'failed' || job.status === 'cancelled') {
            this.queryRunning.set(false);
            this.queryError.set(job.error_message || 'The query job did not complete.');
            clearActiveQuery();
          }
        },
        error: (error: unknown) => {
          this.queryRunning.set(false);
          this.queryError.set(toApiErrorMessage(error));
        },
      });
  }

  private loadQueryResult(queryRunId: string): void {
    if (this.resultLoading) {
      return;
    }
    this.resultLoading = true;
    this.queriesApi
      .getQueryRun(queryRunId)
      .pipe(
        retry({ count: 2, delay: 1_000 }),
        finalize(() => {
          this.resultLoading = false;
          this.queryRunning.set(false);
        }),
      )
      .subscribe({
        next: (result) => {
          this.queryResult.set(result);
          clearActiveQuery();
        },
        error: (error: unknown) => this.queryError.set(toApiErrorMessage(error)),
      });
  }

  private resumeActiveQuery(): void {
    const active = readActiveQuery();
    if (active === null) {
      return;
    }
    this.queryRunning.set(true);
    this.trackQueryJob(active.jobId, active.queryRunId);
  }

  private stopTracking(): void {
    this.jobPolling?.unsubscribe();
    this.jobPolling = null;
    this.resultLoading = false;
    clearActiveQuery();
  }

  private loadPipelines(): void {
    this.pipelinesLoading.set(true);
    this.pipelinesError.set(null);

    this.queriesApi
      .listPipelines()
      .pipe(finalize(() => this.pipelinesLoading.set(false)))
      .subscribe({
        next: (result) => {
          this.pipelines.set(result.pipelines);
          this.defaultPipelineName.set(result.default_pipeline_name);
        },
        error: (error: unknown) => {
          this.pipelinesError.set(toApiErrorMessage(error));
        },
      });
  }

  private loadSubjects(): void {
    const requestId = ++this.subjectsRequest;
    this.subjectsLoading.set(true);
    this.subjectsError.set(null);
    this.subjectsApi.listSubjects().pipe(
      finalize(() => {
        if (requestId === this.subjectsRequest) {
          this.subjectsLoading.set(false);
        }
      }),
    ).subscribe({
      next: (subjects) => {
        if (requestId === this.subjectsRequest) {
          this.subjects.set(subjects);
        }
      },
      error: (error: unknown) => {
        if (requestId === this.subjectsRequest) {
          this.subjectsError.set(toApiErrorMessage(error));
        }
      },
    });
  }
}

const ACTIVE_QUERY_STORAGE_KEY = 'indexer.active-query-job';

interface ActiveQueryReference {
  jobId: string;
  queryRunId: string;
}

function isTerminal(job: BackgroundJob): boolean {
  return job.status === 'succeeded' || job.status === 'failed' || job.status === 'cancelled';
}

function storeActiveQuery(jobId: string, queryRunId: string): void {
  localStorage.setItem(ACTIVE_QUERY_STORAGE_KEY, JSON.stringify({ jobId, queryRunId }));
}

function readActiveQuery(): ActiveQueryReference | null {
  const stored = localStorage.getItem(ACTIVE_QUERY_STORAGE_KEY);
  if (stored === null) {
    return null;
  }
  try {
    const parsed = JSON.parse(stored) as Partial<ActiveQueryReference>;
    if (typeof parsed.jobId === 'string' && typeof parsed.queryRunId === 'string') {
      return { jobId: parsed.jobId, queryRunId: parsed.queryRunId };
    }
    clearActiveQuery();
    return null;
  } catch {
    clearActiveQuery();
    return null;
  }
}

function clearActiveQuery(): void {
  localStorage.removeItem(ACTIVE_QUERY_STORAGE_KEY);
}
