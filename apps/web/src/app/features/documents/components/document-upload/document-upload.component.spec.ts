import { DocumentUploadComponent } from './document-upload.component';

describe('DocumentUploadComponent', () => {
  it('uploads without exposing legacy subject selection', () => {
    const component = new DocumentUploadComponent();
    const emitted: unknown[] = [];
    component.uploadRequested.subscribe((request) => emitted.push(request));
    component.selectedFiles.set([
      new File(['content'], 'notes.md', { type: 'text/markdown' }),
    ]);
    component.submitUpload();

    expect(emitted).toEqual([
      expect.objectContaining({ subjectIds: [] }),
    ]);
  });
});
