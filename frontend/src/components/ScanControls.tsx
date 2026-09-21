import React, { useState } from 'react';
import { FolderSearch, Play, RefreshCw } from 'lucide-react';

interface ScanControlsProps {
  onTriggerScan: (
    customNodesPath?: string,
    workflowsPath?: string,
  ) => Promise<void>;
  isLoading: boolean;
}

export const ScanControls: React.FC<ScanControlsProps> = ({
  onTriggerScan,
  isLoading,
}) => {
  const [customNodesPath, setCustomNodesPath] = useState('');
  const [workflowsPath, setWorkflowsPath] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    void onTriggerScan(
      customNodesPath.trim() || undefined,
      workflowsPath.trim() || undefined,
    );
  };

  return (
    <div className="bg-surface-900 border border-slate-800 rounded-xl p-5 shadow-lg">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FolderSearch className="w-5 h-5 text-brand-500" />
            <h2 className="text-sm font-semibold text-slate-200">
              Execution Controls
            </h2>
          </div>

          <button
            type="button"
            onClick={() => setShowAdvanced((visible) => !visible)}
            className="text-xs text-slate-400 hover:text-slate-200 underline"
          >
            {showAdvanced
              ? 'Hide Custom Roots'
              : 'Configure Custom Search Roots'}
          </button>
        </div>

        {showAdvanced && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Custom Nodes Path (Override)
              </label>

              <input
                type="text"
                placeholder="e.g. C:\ComfyUI\custom_nodes or /opt/comfy/custom_nodes"
                value={customNodesPath}
                onChange={(event) =>
                  setCustomNodesPath(event.target.value)
                }
                className="w-full bg-surface-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Workflows Scan Directory (Override)
              </label>

              <input
                type="text"
                placeholder="e.g. C:\ComfyUI\output or /home/user/workflows"
                value={workflowsPath}
                onChange={(event) =>
                  setWorkflowsPath(event.target.value)
                }
                className="w-full bg-surface-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500"
              />
            </div>
          </div>
        )}

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={isLoading}
            className="flex items-center space-x-2 bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Auditing File System...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Execute Complete Audit Scan</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
