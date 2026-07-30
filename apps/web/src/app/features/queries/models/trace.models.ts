export interface TraceStep {
  id: string | null;
  step_order: number;
  name: string;
  step_type: string | null;
  status: string;
  duration_ms: number | null;
  input_summary: string | null;
  output_summary: string | null;
  error_message: string | null;
  metadata: Record<string, unknown>;
}
