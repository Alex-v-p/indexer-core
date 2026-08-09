import { NgFor, NgIf } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Observable, finalize, forkJoin } from 'rxjs';

import { toApiErrorMessage } from '../../../../core/http/api-error';
import { DocumentOrganizationApiService } from '../../data-access/document-organization-api.service';
import { ContentGroup, DocumentType } from '../../models/document-organization.models';

@Component({
  selector: 'app-organization-page',
  standalone: true,
  imports: [FormsModule, NgFor, NgIf],
  templateUrl: './organization-page.component.html',
})
export class OrganizationPageComponent implements OnInit {
  private readonly organizationApi = inject(DocumentOrganizationApiService);

  readonly contentGroups = signal<ContentGroup[]>([]);
  readonly documentTypes = signal<DocumentType[]>([]);
  readonly selectedGroup = signal<ContentGroup | null>(null);
  readonly selectedType = signal<DocumentType | null>(null);
  readonly loading = signal(false);
  readonly mutationBusy = signal(false);
  readonly backfillBusy = signal(false);
  readonly message = signal<string | null>(null);
  readonly error = signal<string | null>(null);

  groupName = '';
  groupDescription = '';
  groupRename = '';
  aliasName = '';
  typeKey = '';
  typeLabel = '';
  typeDescription = '';
  editTypeLabel = '';
  editTypeDescription = '';
  backfillLimit = 100;

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.error.set(null);
    forkJoin({
      groups: this.organizationApi.listContentGroups(),
      types: this.organizationApi.listDocumentTypes(),
    }).pipe(finalize(() => this.loading.set(false))).subscribe({
      next: ({ groups, types }) => {
        this.contentGroups.set(groups);
        this.documentTypes.set(types);
        const selectedGroupId = this.selectedGroup()?.id;
        const group = groups.find((item) => item.id === selectedGroupId) ?? null;
        this.selectedGroup.set(group);
        if (group) {
          this.groupRename = group.name;
        }
        const selectedTypeId = this.selectedType()?.id;
        const type = types.find((item) => item.id === selectedTypeId) ?? null;
        this.selectedType.set(type);
        if (type) {
          this.setTypeEditor(type);
        }
      },
      error: (error: unknown) => this.error.set(toApiErrorMessage(error)),
    });
  }

  createGroup(): void {
    const name = this.groupName.trim();
    if (!name) {
      return;
    }
    this.runMutation(
      this.organizationApi.createContentGroup({
        name,
        description: this.groupDescription.trim() || undefined,
      }),
      (group) => {
        this.groupName = '';
        this.groupDescription = '';
        this.contentGroups.update((items) => [...items, group]);
        this.selectGroup(group);
      },
    );
  }

  selectGroup(group: ContentGroup): void {
    this.selectedGroup.set(group);
    this.groupRename = group.name;
    this.aliasName = '';
  }

  renameGroup(): void {
    const group = this.selectedGroup();
    const name = this.groupRename.trim();
    if (!group || !name || name === group.name) {
      return;
    }
    this.runMutation(this.organizationApi.updateContentGroup(group.id, { name }), (updated) => {
      this.replaceGroup(updated);
      this.selectedGroup.set(updated);
    });
  }

  archiveGroup(): void {
    const group = this.selectedGroup();
    if (!group || !window.confirm(`Archive “${group.name}”? Existing history is retained.`)) {
      return;
    }
    this.runMutation(this.organizationApi.updateContentGroup(group.id, { archive: true }), () => {
      this.contentGroups.update((items) => items.filter((item) => item.id !== group.id));
      this.selectedGroup.set(null);
    });
  }

  addAlias(): void {
    const group = this.selectedGroup();
    const name = this.aliasName.trim();
    if (!group || !name) {
      return;
    }
    this.runMutation(this.organizationApi.addContentGroupAlias(group.id, name), (alias) => {
      const updated = { ...group, aliases: [...group.aliases, alias] };
      this.aliasName = '';
      this.replaceGroup(updated);
      this.selectedGroup.set(updated);
    });
  }

  archiveAlias(aliasId: string): void {
    const group = this.selectedGroup();
    if (!group) {
      return;
    }
    this.runMutation(this.organizationApi.archiveContentGroupAlias(group.id, aliasId), () => {
      const updated = {
        ...group,
        aliases: group.aliases.filter((alias) => alias.id !== aliasId),
      };
      this.replaceGroup(updated);
      this.selectedGroup.set(updated);
    });
  }

  createType(): void {
    const key = this.typeKey.trim();
    const label = this.typeLabel.trim();
    if (!key || !label) {
      return;
    }
    this.runMutation(
      this.organizationApi.createDocumentType({
        key,
        label,
        description: this.typeDescription.trim() || undefined,
      }),
      (type) => {
        this.typeKey = '';
        this.typeLabel = '';
        this.typeDescription = '';
        this.documentTypes.update((items) => [...items, type]);
        this.selectType(type);
      },
    );
  }

  selectType(type: DocumentType): void {
    this.selectedType.set(type);
    this.setTypeEditor(type);
  }

  updateType(): void {
    const type = this.selectedType();
    const label = this.editTypeLabel.trim();
    if (!type || !label) {
      return;
    }
    this.runMutation(
      this.organizationApi.updateDocumentType(type.id, {
        label,
        description: this.editTypeDescription.trim(),
      }),
      (updated) => {
        this.replaceType(updated);
        this.selectedType.set(updated);
        this.setTypeEditor(updated);
      },
    );
  }

  archiveType(): void {
    const type = this.selectedType();
    if (!type || !window.confirm(`Archive “${type.label}”? Existing decisions are retained.`)) {
      return;
    }
    this.runMutation(this.organizationApi.updateDocumentType(type.id, { archive: true }), () => {
      this.documentTypes.update((items) => items.filter((item) => item.id !== type.id));
      this.selectedType.set(null);
    });
  }

  startBackfill(): void {
    if (this.backfillBusy()) {
      return;
    }
    const limit = Math.min(5000, Math.max(1, Math.trunc(Number(this.backfillLimit) || 100)));
    this.backfillLimit = limit;
    this.backfillBusy.set(true);
    this.error.set(null);
    this.message.set(null);
    this.organizationApi.backfill(limit).pipe(
      finalize(() => this.backfillBusy.set(false)),
    ).subscribe({
      next: (result) => this.message.set(
        result.job_ids.length > 0
          ? `Queued ${result.job_ids.length} organization job(s); ${result.skipped_document_ids.length} document(s) were skipped.`
          : `No jobs were needed; ${result.skipped_document_ids.length} document(s) were skipped.`,
      ),
      error: (error: unknown) => this.error.set(toApiErrorMessage(error)),
    });
  }

  private replaceGroup(updated: ContentGroup): void {
    this.contentGroups.update((items) =>
      items.map((item) => item.id === updated.id ? updated : item),
    );
  }

  private replaceType(updated: DocumentType): void {
    this.documentTypes.update((items) =>
      items.map((item) => item.id === updated.id ? updated : item),
    );
  }

  private setTypeEditor(type: DocumentType): void {
    this.editTypeLabel = type.label;
    this.editTypeDescription = type.description ?? '';
  }

  private runMutation<T>(request: Observable<T>, onSuccess: (value: T) => void): void {
    this.mutationBusy.set(true);
    this.error.set(null);
    this.message.set(null);
    request.pipe(finalize(() => this.mutationBusy.set(false))).subscribe({
      next: onSuccess,
      error: (error: unknown) => this.error.set(toApiErrorMessage(error)),
    });
  }
}
