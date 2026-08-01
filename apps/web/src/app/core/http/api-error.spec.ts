import { HttpErrorResponse } from '@angular/common/http';

import { toApiErrorMessage } from './api-error';

describe('toApiErrorMessage', () => {
  it('surfaces structured conflict messages', () => {
    const error = new HttpErrorResponse({
      status: 409,
      statusText: 'Conflict',
      error: {
        detail: {
          message: 'The subject assignment changed. Refresh and try again.',
          current: null,
        },
      },
    });

    expect(toApiErrorMessage(error)).toBe(
      'The subject assignment changed. Refresh and try again.',
    );
  });
});
