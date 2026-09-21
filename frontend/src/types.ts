export type PackageStatus = 'USED' | 'UNUSED' | 'UNKNOWN';

export type ScanStatus =
  | 'PENDING'
  | 'RUNNING'
  | 'COMPLETED'
  | 'FAILED';

export interface NodeItem {
  id: number;
  node_name: string;
  node_class: string;
  node_type: string;
  referenced_in_workflow: boolean;
  reference_count: number;
  source_snippet: string | null;
}

export interface DiscoveredPackage {
  id: number;
  scan_id: string;
  package_name: string;
  directory_path: string;
  status: PackageStatus;
  detected_nodes_count: number;
  used_nodes_count: number;
  nodes: NodeItem[];
}

export interface WorkflowItem {
  id: number;
  scan_id: string;
  file_path: string;
  referenced_nodes: { type: string; count: number }[];
  image_paths: string[];
}

export interface AuditScanSummary {
  id: string;
  status: ScanStatus;
  initiated_by: string;
  scanned_root_path: string;
  created_at: string;
  completed_at: string | null;
  error_detail: string | null;
}

export interface AuditScanDetail extends AuditScanSummary {
  packages: DiscoveredPackage[];
  workflows: WorkflowItem[];
}
