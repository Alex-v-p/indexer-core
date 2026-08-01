import { NgFor, NgIf } from '@angular/common';
import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Observable, Subscription, finalize, forkJoin, switchMap, takeWhile, timer } from 'rxjs';

import { toApiErrorMessage } from '../../../../core/http/api-error';
import { DocumentsApiService } from '../../../documents/data-access/documents-api.service';
import { DocumentSummary } from '../../../documents/models/document.models';
import { BackgroundJobsApiService } from '../../../../core/background-jobs/background-jobs-api.service';
import { BackgroundJob } from '../../../../core/background-jobs/background-job.models';
import { DocumentJobProgressComponent } from '../../../documents/components/document-job-progress/document-job-progress.component';
import { SubjectsApiService } from '../../data-access/subjects-api.service';
import {
  DocumentSubjectDecision,
  Subject,
  SubjectKind,
} from '../../models/subject.models';

interface SubjectDocumentMembership {
  document: DocumentSummary;
  decision: DocumentSubjectDecision | null;
}

@Component({
  selector: 'app-subjects-page',
  standalone: true,
  imports: [FormsModule, NgFor, NgIf, DocumentJobProgressComponent],
  templateUrl: './subjects-page.component.html',
})
export class SubjectsPageComponent implements OnInit, OnDestroy {
  private readonly subjectsApi = inject(SubjectsApiService);
  private readonly documentsApi = inject(DocumentsApiService);
  private readonly jobsApi = inject(BackgroundJobsApiService);
  private readonly classificationPolling = new Map<string, Subscription>();

  readonly subjects = signal<Subject[]>([]);
  readonly documents = signal<DocumentSummary[]>([]);
  readonly selectedSubject = signal<Subject | null>(null);
  readonly memberships = signal<SubjectDocumentMembership[]>([]);
  readonly loading = signal(false);
  readonly membershipsLoading = signal(false);
  readonly mutationBusy = signal(false);
  readonly error = signal<string | null>(null);
  readonly backfillJobs = signal<BackgroundJob[]>([]);
  readonly backfillBusy = signal(false);
  readonly backfillMessage = signal<string | null>(null);
  private membershipsRequest = 0;

  createName = '';
  createDescription = '';
  createKind: SubjectKind = 'project';
  renameName = '';
  aliasName = '';
  backfillLimit = 25;

  ngOnInit(): void {
    this.refreshWorkspace();
  }

  ngOnDestroy(): void {
    for (const subscription of this.classificationPolling.values()) {
      subscription.unsubscribe();
    }
  }

  startClassificationBackfill(): void {
    if (this.backfillBusy()) {
      return;
    }
    const limit = Math.min(100, Math.max(1, Math.trunc(Number(this.backfillLimit) || 25)));
    this.backfillLimit = limit;
    this.backfillBusy.set(true);
    this.backfillMessage.set(null);
    this.error.set(null);
    this.documentsApi.backfillSubjectClassification(limit).pipe(
      finalize(() => this.backfillBusy.set(false)),
    ).subscribe({
      next: (result) => {
        this.backfillMessage.set(
          result.jobs.length > 0
            ? `Queued ${result.jobs.length} classification job(s); ${result.skipped_document_ids.length} document(s) were already current.`
            : `No classification jobs were needed; ${result.skipped_document_ids.length} document(s) were already current.`,
        );
        for (const queued of result.jobs) {
          this.trackClassificationJob(queued.job_id);
        }
      },
      error: (error: unknown) => this.error.set(toApiErrorMessage(error)),
    });
  }

  refreshWorkspace(): void {
    this.loading.set(true);
    this.error.set(null);
    forkJoin({
      subjects: this.subjectsApi.listSubjects(),
      documents: this.documentsApi.listDocuments(),
    })
      .pipe(finalize(() => this.loading.set(false)))
      .subscribe({
        next: ({ subjects, documents }) => {
          this.subjects.set(subjects);
          this.documents.set(documents);
          const selectedId = this.selectedSubject()?.id;
          const selected = subjects.find((subject) => subject.id === selectedId) ?? null;
          this.selectedSubject.set(selected);
          if (selected) {
            this.renameName = selected.name;
            this.loadMemberships(selected.id);
          }
        },
        error: (error: unknown) => this.error.set(toApiErrorMessage(error)),
      });
  }

  createSubject(): void {
    const name = this.createName.trim();
    if (!name) {
      return;
    }
    this.runMutation(
      this.subjectsApi.createSubject({
        kind: this.createKind,
        name,
        description: this.createDescription.trim() || undefined,
      }),
      (subject) => {
        this.createName = '';
        this.createDescription = '';
        this.subjects.update((subjects) => [...subjects, subject]);
        this.selectSubject(subject);
      },
    );
  }

  selectSubject(subject: Subject): void {
    this.selectedSubject.set(subject);
    this.memberships.set([]);
    this.renameName = subject.name;
    this.aliasName = '';
    this.loadMemberships(subject.id);
  }

  renameSubject(): void {
    const subject = this.selectedSubject();
    const name = this.renameName.trim();
    if (!subject || !name || name === subject.name) {
      return;
    }
    this.runMutation(this.subjectsApi.updateSubject(subject.id, { name }), (updated) => {
      this.replaceSubject(updated);
      if (this.selectedSubject()?.id === subject.id) {
        this.selectedSubject.set(updated);
      }
    }, undefined, () => this.selectedSubject()?.id === subject.id);
  }

  archiveSubject(): void {
    const subject = this.selectedSubject();
    if (!subject || !window.confirm(`Archive “${subject.name}”? Existing history is retained.`)) {
      return;
    }
    this.runMutation(this.subjectsApi.updateSubject(subject.id, { archive: true }), () => {
      this.subjects.update((subjects) => subjects.filter((item) => item.id !== subject.id));
      this.selectedSubject.set(null);
      this.memberships.set([]);
    }, undefined, () => this.selectedSubject()?.id === subject.id);
  }

  addAlias(): void {
    const subject = this.selectedSubject();
    const name = this.aliasName.trim();
    if (!subject || !name) {
      return;
    }
    this.runMutation(this.subjectsApi.addAlias(subject.id, name), (alias) => {
      const updated = { ...subject, aliases: [...subject.aliases, alias] };
      this.aliasName = '';
      this.replaceSubject(updated);
      this.selectedSubject.set(updated);
    }, undefined, () => this.selectedSubject()?.id === subject.id);
  }

  archiveAlias(aliasId: string): void {
    const subject = this.selectedSubject();
    if (!subject) {
      return;
    }
    this.runMutation(this.subjectsApi.archiveAlias(subject.id, aliasId), () => {
      const updated = {
        ...subject,
        aliases: subject.aliases.filter((alias) => alias.id !== aliasId),
      };
      this.replaceSubject(updated);
      this.selectedSubject.set(updated);
    }, undefined, () => this.selectedSubject()?.id === subject.id);
  }

  assignMembership(row: SubjectDocumentMembership): void {
    this.setMembership(row, 'assigned');
  }

  rejectMembership(row: SubjectDocumentMembership): void {
    this.setMembership(row, 'rejected');
  }

  reviewSuggestion(row: SubjectDocumentMembership, review: 'accept' | 'reject'): void {
    const subject = this.selectedSubject();
    if (!subject || !row.decision) {
      return;
    }
    this.runMutation(
      this.subjectsApi.reviewSuggestion(
        row.document.id,
        subject.id,
        review,
        row.decision.revision,
      ),
      () => this.loadMemberships(subject.id),
      () => this.loadMemberships(subject.id),
      () => this.selectedSubject()?.id === subject.id,
    );
  }

  private setMembership(
    row: SubjectDocumentMembership,
    state: 'assigned' | 'rejected',
  ): void {
    const subject = this.selectedSubject();
    if (!subject) {
      return;
    }
    this.runMutation(
      this.subjectsApi.setDocumentDecision(
        row.document.id,
        subject.id,
        state,
        row.decision?.revision ?? 0,
      ),
      () => this.loadMemberships(subject.id),
      () => this.loadMemberships(subject.id),
      () => this.selectedSubject()?.id === subject.id,
    );
  }

  private loadMemberships(subjectId: string): void {
    const requestId = ++this.membershipsRequest;
    const documents = this.documents();
    this.memberships.set([]);
    this.membershipsLoading.set(true);
    if (documents.length === 0) {
      this.membershipsLoading.set(false);
      return;
    }
    forkJoin(
      documents.map((document) =>
        this.subjectsApi.listDocumentDecisions(document.id),
      ),
    ).pipe(
      finalize(() => {
        if (
          this.selectedSubject()?.id === subjectId &&
          this.membershipsRequest === requestId
        ) {
          this.membershipsLoading.set(false);
        }
      }),
    ).subscribe({
      next: (decisionSets) => {
        if (
          this.selectedSubject()?.id !== subjectId ||
          this.membershipsRequest !== requestId
        ) {
          return;
        }
        this.memberships.set(
          documents.map((document, index) => ({
            document,
            decision:
              decisionSets[index].find((decision) => decision.subject_id === subjectId) ??
              null,
          })),
        );
      },
      error: (error: unknown) => {
        if (
          this.selectedSubject()?.id === subjectId &&
          this.membershipsRequest === requestId
        ) {
          this.error.set(toApiErrorMessage(error));
        }
      },
    });
  }

  private replaceSubject(updated: Subject): void {
    this.subjects.update((subjects) =>
      subjects.map((subject) => (subject.id === updated.id ? updated : subject)),
    );
  }

  private trackClassificationJob(jobId: string): void {
    if (this.classificationPolling.has(jobId)) {
      return;
    }
    const subscription = timer(0, 1000).pipe(
      switchMap(() => this.jobsApi.getJob(jobId)),
      takeWhile((job) => !isTerminalJob(job), true),
    ).subscribe({
      next: (job) => {
        this.backfillJobs.update((jobs) => [job, ...jobs.filter((item) => item.id !== job.id)].slice(0, 25));
        if (isTerminalJob(job)) {
          this.classificationPolling.delete(job.id);
          if (job.status === 'failed') {
            this.error.set(job.error_message || 'A subject classification job failed.');
          }
          if (this.classificationPolling.size === 0) {
            this.refreshWorkspace();
          }
        }
      },
      error: (error: unknown) => {
        this.classificationPolling.delete(jobId);
        this.error.set(`A classification job could not be refreshed: ${toApiErrorMessage(error)}`);
      },
    });
    this.classificationPolling.set(jobId, subscription);
  }

  private runMutation<T>(
    request: Observable<T>,
    onSuccess: (value: T) => void,
    onError?: () => void,
    isRelevant: () => boolean = () => true,
  ): void {
    this.mutationBusy.set(true);
    this.error.set(null);
    request.pipe(finalize(() => this.mutationBusy.set(false))).subscribe({
      next: (value) => {
        if (isRelevant()) {
          onSuccess(value);
        }
      },
      error: (error: unknown) => {
        if (isRelevant()) {
          this.error.set(toApiErrorMessage(error));
          onError?.();
        }
      },
    });
  }
}

function isTerminalJob(job: BackgroundJob): boolean {
  return job.status === 'succeeded' || job.status === 'failed' || job.status === 'cancelled';
}
