import { DecimalPipe, JsonPipe, NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { metadataString } from '../../../../shared/utils/formatting';
import { EvidenceItem } from '../../models/query.models';

@Component({
  selector: 'app-evidence-viewer',
  standalone: true,
  imports: [DecimalPipe, JsonPipe, NgFor, NgIf],
  templateUrl: './evidence-viewer.component.html',
})
export class EvidenceViewerComponent {
  @Input() evidence: EvidenceItem[] = [];

  trackEvidence(index: number, item: EvidenceItem): string {
    return item.id ?? `${item.rank}-${index}`;
  }

  evidenceSource(item: EvidenceItem): string {
    const filename = metadataString(item.metadata, 'original_filename');
    const section = metadataString(item.metadata, 'section_title');
    return [filename, section].filter(Boolean).join(' · ') || 'Retrieved source chunk';
  }
}
