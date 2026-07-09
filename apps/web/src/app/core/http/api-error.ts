import { HttpErrorResponse } from '@angular/common/http';

export function toApiErrorMessage(error: unknown): string {
  if (error instanceof HttpErrorResponse) {
    const detail = error.error?.detail;

    if (typeof detail === 'string' && detail.trim().length > 0) {
      return detail;
    }

    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item: unknown) => {
          if (typeof item === 'object' && item !== null && 'msg' in item) {
            return String((item as { msg: unknown }).msg);
          }
          return String(item);
        })
        .join(', ');
    }

    if (error.status > 0) {
      return `${error.status} ${error.statusText || 'API error'}`.trim();
    }
  }

  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }

  return 'Something went wrong while contacting the API.';
}
