import { DecimalPipe, JsonPipe, NgClass, NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { metadataNumber, metadataString } from '../../../../shared/utils/formatting';
import { EvidenceGrade, EvidenceGrading, EvidenceItem } from '../../models/query.models';

@Component({
  selector: 'app-evidence-viewer',
  standalone: true,
  imports: [DecimalPipe, JsonPipe, NgClass, NgFor, NgIf],
  templateUrl: './evidence-viewer.component.html',
})
export class EvidenceViewerComponent {
  @Input() evidence: EvidenceItem[] = [];
  @Input() grading: EvidenceGrading | null = null;

  trackEvidence(index: number, item: EvidenceItem): string {
    return item.id ?? `${item.rank}-${index}`;
  }

  laneName(item: EvidenceItem): string | null {
    return item.subject_name ?? metadataString(item.metadata, 'subject_name');
  }

  evidenceSource(item: EvidenceItem): string {
    const filename = metadataString(item.metadata, 'original_filename');
    const versionLabel = metadataString(item.metadata, 'document_version_label');
    const versionNumber = metadataNumber(item.metadata, 'document_version_number');
    const version = versionLabel ?? (versionNumber === null ? null : `v${versionNumber}`);
    const section = metadataString(item.metadata, 'section_title');
    return [filename, version, section].filter(Boolean).join(' · ') || 'Retrieved source chunk';
  }

  evidenceGrade(item: EvidenceItem): EvidenceGrade | null {
    return this.grading?.grades.find((grade) => grade.evidence_rank === item.rank) ?? null;
  }

  percentage(value: number): number {
    return Math.round(Math.min(Math.max(value, 0), 1) * 100);
  }
}
