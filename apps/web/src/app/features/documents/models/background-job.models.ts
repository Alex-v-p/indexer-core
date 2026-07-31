export type BackgroundJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface BackgroundJob {
  id: string;
  job_type: string;
  status: BackgroundJobStatus;
  priority: number;
  payload: Record<string, unknown>;
  result: Record<string, unknown>;
  progress: number;
  current_stage: string | null;
  attempts: number;
  max_attempts: number;
  dedupe_key: string | null;
  scheduled_at: string;
  heartbeat_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface QueuedDocumentOperation {
  job_id: string;
  status: string;
}
