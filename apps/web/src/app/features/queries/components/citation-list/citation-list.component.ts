import { NgFor, NgIf } from '@angular/common';
import { Component, Input } from '@angular/core';

import { metadataNumber, metadataString } from '../../../../shared/utils/formatting';
import { CitationItem } from '../../models/query.models';

@Component({
  selector: 'app-citation-list',
  standalone: true,
  imports: [NgFor, NgIf],
  templateUrl: './citation-list.component.html',
})
export class CitationListComponent {
  @Input() citations: CitationItem[] = [];

  trackCitation(index: number, citation: CitationItem): string {
    return citation.id ?? `${citation.citation_index}-${index}`;
  }

  citationSource(citation: CitationItem): string {
    const filename = metadataString(citation.metadata, 'original_filename');
    const versionLabel = metadataString(citation.metadata, 'document_version_label');
    const versionNumber = metadataNumber(citation.metadata, 'document_version_number');
    const version = versionLabel ?? (versionNumber === null ? null : `v${versionNumber}`);
    const section = metadataString(citation.metadata, 'section_title');
    return [filename, version, section].filter(Boolean).join(' · ') || 'Retrieved source chunk';
  }
}
