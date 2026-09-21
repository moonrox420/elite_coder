import { AuditScanDetail } from '../types';

const API_BASE = '/api/v1';

export class ApiService {
  private static token: string | null = null;

  static setToken(token: string): void {
    this.token = token;
  }

  private static getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    return headers;
  }

  static async triggerScan(
    paths?: {
      customNodes?: string[];
      workflows?: string[];
    },
  ): Promise<{ id: string }> {
    const response = await fetch(`${API_BASE}/scans/`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        custom_nodes_roots: paths?.customNodes ?? [],
        workflows_roots: paths?.workflows ?? [],
        include_image_metadata: true,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to initiate audit scan');
    }

    return response.json();
  }

  static async getScanDetails(
    scanId: string,
  ): Promise<AuditScanDetail> {
    const response = await fetch(
      `${API_BASE}/scans/${encodeURIComponent(scanId)}`,
      {
        headers: this.getHeaders(),
      },
    );

    if (!response.ok) {
      throw new Error('Failed to fetch scan details');
    }

    return response.json();
  }

  static async listRecentScans(): Promise<AuditScanDetail[]> {
    const response = await fetch(`${API_BASE}/scans/`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch scans list');
    }

    return response.json();
  }
}
