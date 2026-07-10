export interface DocumentSummary {
  id: string;
  title: string;
  original_filename: string | null;
  content_type: string | null;
  storage_uri: string | null;
  size_bytes: number | null;
  checksum_sha256: string | null;
  status: string;
  chunk_count: number;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface DocumentVersion {
  id: string;
  version_number: number;
  storage_uri: string | null;
  content_type: string | null;
  checksum_sha256: string | null;
  parser_name: string | null;
  parser_version: string | null;
  status: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ChunkIndex {
  id: string;
  ordinal: number;
  content_hash: string | null;
  token_count: number | null;
  source_page_start: number | null;
  source_page_end: number | null;
  section_title: string | null;
  qdrant_collection: string;
  qdrant_point_id: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface DocumentDetail extends DocumentSummary {
  versions: DocumentVersion[];
  chunks: ChunkIndex[];
}
