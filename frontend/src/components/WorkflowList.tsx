import React, { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, FileJson, Image } from 'lucide-react';
import { WorkflowItem } from '../types';

interface WorkflowListProps {
  workflows: WorkflowItem[];
}

export const WorkflowList: React.FC<WorkflowListProps> = ({
  workflows,
}) => {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggle = (id: number) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const sorted = useMemo(
    () => [...workflows].sort((a, b) =>
      a.file_path.localeCompare(b.file_path),
    ),
    [workflows],
  );

  return (
    <div className="bg-surface-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <FileJson className="w-4 h-4 text-brand-500" />
          <h2 className="text-sm font-semibold text-slate-200">
            Analyzed Workflows
          </h2>
        </div>

        <span className="text-[11px] text-slate-400 font-mono">
          {workflows.length}
        </span>
      </div>

      {sorted.length === 0 ? (
        <div className="p-8 text-center">
          <p className="text-xs text-slate-500">
            No workflow files discovered in the scan scope.
          </p>
        </div>
      ) : (
        <ul className="divide-y divide-slate-800 max-h-[520px] overflow-y-auto">
          {sorted.map((workflow) => {
            const isOpen = expanded.has(workflow.id);
            const fileName =
              workflow.file_path.split(/[\\/]/).pop() ??
              workflow.file_path;

            return (
              <li key={workflow.id}>
                <button
                  type="button"
                  onClick={() => toggle(workflow.id)}
                  className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-800/40 transition"
                >
                  <span className="flex items-center space-x-2 min-w-0">
                    {isOpen ? (
                      <ChevronDown className="w-4 h-4 text-slate-500 shrink-0" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-slate-500 shrink-0" />
                    )}
                    <span className="text-xs font-medium text-slate-200 truncate">
                      {fileName}
                    </span>
                  </span>

                  <span className="flex items-center space-x-2 shrink-0">
                    {workflow.image_paths.length > 0 && (
                      <span
                        className="inline-flex items-center text-[10px] text-slate-500"
                        title={`${workflow.image_paths.length} image asset(s)`}
                      >
                        <Image className="w-3 h-3 mr-1" />
                        {workflow.image_paths.length}
                      </span>
                    )}
                    <span className="text-[10px] font-mono text-slate-500">
                      {workflow.referenced_nodes.length} refs
                    </span>
                  </span>
                </button>

                {isOpen && (
                  <div className="px-4 pb-4">
                    <p
                      className="text-[11px] font-mono text-slate-500 truncate mb-2"
                      title={workflow.file_path}
                    >
                      {workflow.file_path}
                    </p>

                    <div className="flex flex-wrap gap-1.5">
                      {workflow.referenced_nodes.map((ref) => (
                        <span
                          key={ref.type}
                          className="inline-flex items-center px-2 py-0.5 rounded-md bg-surface-950 border border-slate-800 text-[10px] font-mono text-slate-300"
                        >
                          {ref.type}
                          <span className="ml-1.5 text-slate-500">
                            ×{ref.count}
                          </span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};
