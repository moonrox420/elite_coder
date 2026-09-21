import React, { useCallback, useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { MetricsCards } from './components/MetricsCards';
import { ScanControls } from './components/ScanControls';
import { PackageTable } from './components/PackageTable';
import { WorkflowList } from './components/WorkflowList';
import { PackageDetailModal } from './components/PackageDetailModal';
import { ApiService } from './services/api';
import { AuditScanDetail, DiscoveredPackage } from './types';

export const App: React.FC = () => {
  const [currentScan, setCurrentScan] =
    useState<AuditScanDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedPackage, setSelectedPackage] =
    useState<DiscoveredPackage | null>(null);

  const fetchScanDetails = useCallback(
    async (scanId: string): Promise<void> => {
      try {
        const data = await ApiService.getScanDetails(scanId);
        setCurrentScan(data);

        if (
          data.status === 'RUNNING' ||
          data.status === 'PENDING'
        ) {
          window.setTimeout(() => {
            void fetchScanDetails(scanId);
          }, 2000);
          return;
        }

        setIsLoading(false);
      } catch (error) {
        console.error('Failed to fetch scan details:', error);
        setIsLoading(false);
      }
    },
    [],
  );

  const handleTriggerScan = async (
    customNodesPath?: string,
    workflowsPath?: string,
  ): Promise<void> => {
    setIsLoading(true);

    try {
      const result = await ApiService.triggerScan({
        customNodes: customNodesPath
          ? [customNodesPath]
          : undefined,
        workflows: workflowsPath
          ? [workflowsPath]
          : undefined,
      });

      await fetchScanDetails(result.id);
    } catch (error) {
      console.error('Failed to initiate scan:', error);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const loadRecentScan = async (): Promise<void> => {
      try {
        const scans = await ApiService.listRecentScans();

        if (!cancelled && scans.length > 0) {
          await fetchScanDetails(scans[0].id);
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to load recent scans:', error);
        }
      }
    };

    void loadRecentScan();

    return () => {
      cancelled = true;
    };
  }, [fetchScanDetails]);

  return (
    <div className="min-h-screen bg-surface-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-1 p-6 space-y-6 max-w-7xl mx-auto w-full">
        <MetricsCards scan={currentScan} />

        <ScanControls
          onTriggerScan={handleTriggerScan}
          isLoading={isLoading}
        />

        {currentScan && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PackageTable
                packages={currentScan.packages ?? []}
                onSelectPackage={setSelectedPackage}
              />
            </div>

            <div>
              <WorkflowList
                workflows={currentScan.workflows ?? []}
              />
            </div>
          </div>
        )}
      </main>

      <PackageDetailModal
        pkg={selectedPackage}
        onClose={() => setSelectedPackage(null)}
      />
    </div>
  );
};

export default App;
