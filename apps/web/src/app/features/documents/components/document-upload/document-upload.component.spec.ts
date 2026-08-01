import { DocumentUploadComponent } from './document-upload.component';

describe('DocumentUploadComponent', () => {
  it('emits selected subjects and clears them after upload', () => {
    const component = new DocumentUploadComponent();
    const emitted: unknown[] = [];
    component.uploadRequested.subscribe((request) => emitted.push(request));
    component.selectedFiles.set([
      new File(['content'], 'notes.md', { type: 'text/markdown' }),
    ]);
    component.toggleSubject('subject-1', true);
    component.toggleSubject('subject-2', true);

    component.submitUpload();

    expect(emitted).toEqual([
      expect.objectContaining({ subjectIds: ['subject-1', 'subject-2'] }),
    ]);
    expect(component.selectedSubjectIds()).toEqual([]);
  });
});
