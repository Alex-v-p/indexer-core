import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';

import { API_BASE_URL } from '../config/api.config';

export const apiBaseUrlInterceptor: HttpInterceptorFn = (request, next) => {
  const apiBaseUrl = inject(API_BASE_URL).replace(/\/$/, '');

  if (/^https?:\/\//i.test(request.url) || request.url.startsWith('/assets/')) {
    return next(request);
  }

  const normalizedUrl = request.url.startsWith('/') ? request.url : `/${request.url}`;
  return next(request.clone({ url: `${apiBaseUrl}${normalizedUrl}` }));
};
