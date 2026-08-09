import { NgFor, NgIf } from '@angular/common';
import { Component, EventEmitter, Input, Output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  ContentGroup,
  DocumentOrganization,
  DocumentType,
  ManualDocumentTypeDecision,
} from '../../../organization/models/document-organization.models';

export interface DocumentGroupChangeRequest {
  contentGroupId: string | null;
  expectedRevision: number;
}

export interface DocumentTypesChangeRequest {
  decisions: ManualDocumentTypeDecision[];
}

@Component({
  selector: 'app-document-organization-panel',
  standalone: true,
  imports: [FormsModule, NgFor, NgIf],
  templateUrl: './document-organization-panel.component.html',
})
export class DocumentOrganizationPanelComponent {
  @Input() contentGroups: ContentGroup[] = [];
  @Input() documentTypes: DocumentType[] = [];
  @Input() busy = false;
  @Input() loading = false;
  @Input() classificationBusy = false;
  @Output() groupChangeRequested = new EventEmitter<DocumentGroupChangeRequest>();
  @Output() typesChangeRequested = new EventEmitter<DocumentTypesChangeRequest>();
  @Output() reclassifyRequested = new EventEmitter<void>();

  private currentOrganization: DocumentOrganization | null = null;
  readonly selectedTypeIds = signal<string[]>([]);
  selectedGroupId = '';

  @Input()
  set organization(value: DocumentOrganization | null) {
    this.currentOrganization = value;
    this.selectedGroupId = value?.content_group_assignment?.content_group_id ?? '';
    this.selectedTypeIds.set(
      value?.type_decisions
        .filter((item) => item.decision.state === 'assigned')
        .map((item) => item.decision.document_type_id) ?? [],
    );
  }

  get organization(): DocumentOrganization | null {
    return this.currentOrganization;
  }

  get assignmentState(): string {
    return this.organization?.content_group_assignment?.state ?? 'not classified';
  }

  get classificationStatus(): string {
    const value = this.organization?.status?.['status'];
    return typeof value === 'string' ? value : 'not run';
  }

  get classificationError(): string | null {
    const value = this.organization?.status?.['error_message'];
    return typeof value === 'string' ? value : null;
  }

  get classificationIsRunning(): boolean {
    return this.classificationBusy || this.classificationStatus === 'queued' || this.classificationStatus === 'running';
  }

  isTypeSelected(documentTypeId: string): boolean {
    return this.selectedTypeIds().includes(documentTypeId);
  }

  toggleType(documentTypeId: string, selected: boolean): void {
    this.selectedTypeIds.update((ids) => selected
      ? [...new Set([...ids, documentTypeId])]
      : ids.filter((id) => id !== documentTypeId));
  }

  decisionFor(documentTypeId: string) {
    return this.organization?.type_decisions.find(
      (item) => item.decision.document_type_id === documentTypeId,
    )?.decision ?? null;
  }

  saveGroup(): void {
    if (!this.selectedGroupId) {
      return;
    }
    const assignment = this.organization?.content_group_assignment;
    if (assignment?.content_group_id === this.selectedGroupId) {
      return;
    }
    this.groupChangeRequested.emit({
      contentGroupId: this.selectedGroupId,
      expectedRevision: assignment?.revision ?? 0,
    });
  }

  clearGroup(): void {
    const assignment = this.organization?.content_group_assignment;
    this.groupChangeRequested.emit({
      contentGroupId: null,
      expectedRevision: assignment?.revision ?? 0,
    });
  }

  saveTypes(): void {
    const selected = new Set(this.selectedTypeIds());
    const decisions = this.documentTypes.flatMap((type): ManualDocumentTypeDecision[] => {
      const current = this.decisionFor(type.id);
      const currentlyAssigned = current?.state === 'assigned';
      const shouldAssign = selected.has(type.id);
      if (currentlyAssigned === shouldAssign) {
        return [];
      }
      return [{
        document_type_id: type.id,
        state: shouldAssign ? 'assigned' : 'rejected',
        expected_revision: current?.revision ?? 0,
      }];
    });
    if (decisions.length > 0) {
      this.typesChangeRequested.emit({ decisions });
    }
  }

  requestReclassification(): void {
    if (!this.classificationIsRunning) {
      this.reclassifyRequested.emit();
    }
  }
}
