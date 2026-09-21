import React, { useState } from 'react';
import {
  Braces,
  CheckCircle2,
  HelpCircle,
  TerminalSquare,
  X,
} from 'lucide-react';
import { DiscoveredPackage } from '../types';

interface PackageDetailModalProps {
  pkg: DiscoveredPackage | null;
  onClose: () => void;
}

export const PackageDetailModal: React.FC<PackageDetailModalProps> = ({
  pkg,
  onClose,
}) => {
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(
    null,
  );

  if (!pkg) {
    return null;
  }

  const selectedNode =
    pkg.nodes.find((node) => node.id === selectedNodeId) ??
    pkg.nodes[0] ??
    null;

  const statusMeta = {
    USED: {
      icon: CheckCircle2,
      classes:
        'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
    },
    UNUSED: {
      icon: HelpCircle,
      classes:
        'bg-amber-500/10 text-amber-400 border border-amber-500/20',
    },
    UNKNOWN: {
      icon: Braces,
      classes:
        'bg-purple-500/10 text-purple-400 border border-purple-500/20',
    },
  }[pkg.status];

  const StatusIcon = statusMeta.icon;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-surface-900 border border-slate-800 rounded-xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="p-5 border-b border-slate-800 flex items-start justify-between">
          <div className="min-w-0">
            <div className="flex items-center space-x-2">
              <h2 className="text-sm font-bold text-slate-100 truncate">
                {pkg.package_name}
              </h2>
              <span
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${statusMeta.classes}`}
              >
                <StatusIcon className="w-3 h-3" />
                {pkg.status}
              </span>
            </div>

            <p
              className="mt-1 text-[11px] font-mono text-slate-500 truncate"
              title={pkg.directory_path}
            >
              {pkg.directory_path}
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded-md transition"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 grid grid-cols-3 gap-3 border-b border-slate-800">
          <div className="text-center">
            <p className="text-lg font-bold text-slate-100">
              {pkg.detected_nodes_count}
            </p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wide">
              Detected
            </p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-emerald-400">
              {pkg.used_nodes_count}
            </p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wide">
              Referenced
            </p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-slate-100">
              {pkg.nodes.length}
            </p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wide">
              Node Defs
            </p>
          </div>
        </div>

        <div className="p-5 overflow-y-auto space-y-4">
          {pkg.nodes.length === 0 ? (
            <p className="text-xs text-slate-500 text-center py-6">
              No node definitions were detected in this package.
            </p>
          ) : (
            <>
              <div>
                <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide mb-2">
                  Detected Nodes
                </h3>
                <ul className="space-y-1.5">
                  {pkg.nodes.map((node) => (
                    <li key={node.id}>
                      <button
                        type="button"
                        onClick={() => setSelectedNodeId(node.id)}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-lg border text-left transition ${
                          selectedNode?.id === node.id
                            ? 'bg-slate-800/60 border-brand-500/50'
                            : 'bg-surface-950 border-slate-800 hover:border-slate-700'
                        }`}
                      >
                        <span className="flex items-center space-x-2 min-w-0">
                          <TerminalSquare className="w-3.5 h-3.5 text-brand-500 shrink-0" />
                          <span className="text-xs font-mono text-slate-200 truncate">
                            {node.node_type}
                          </span>
                        </span>

                        <span
                          className={`text-[10px] font-medium shrink-0 ${
                            node.referenced_in_workflow
                              ? 'text-emerald-400'
                              : 'text-slate-500'
                          }`}
                        >
                          {node.referenced_in_workflow
                            ? `${node.reference_count} ref`
                            : 'no refs'}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>

              {selectedNode && (
                <div>
                  <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide mb-2">
                    Source AST — {selectedNode.node_class}
                  </h3>

                  <pre className="bg-surface-950 border border-slate-800 rounded-lg p-4 text-[11px] font-mono text-slate-300 overflow-x-auto whitespace-pre-wrap max-h-64 overflow-y-auto">
                    {selectedNode.source_snippet ?? 'Source unavailable.'}
                  </pre>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};
