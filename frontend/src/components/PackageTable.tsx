import React, { useMemo, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Eye,
  HelpCircle,
  Search,
} from 'lucide-react';
import { DiscoveredPackage } from '../types';

interface PackageTableProps {
  packages: DiscoveredPackage[];
  onSelectPackage: (pkg: DiscoveredPackage) => void;
}

type PackageStatus = DiscoveredPackage['status'];

export const PackageTable: React.FC<PackageTableProps> = ({
  packages,
  onSelectPackage,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] =
    useState<PackageStatus | 'ALL'>('ALL');

  const filtered = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase();

    return packages.filter((pkg) => {
      const matchesSearch =
        normalizedSearch.length === 0 ||
        pkg.package_name.toLowerCase().includes(normalizedSearch) ||
        pkg.directory_path.toLowerCase().includes(normalizedSearch);

      const matchesStatus =
        statusFilter === 'ALL' ||
        pkg.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [packages, searchTerm, statusFilter]);

  const getStatusBadge = (status: PackageStatus) => {
    switch (status) {
      case 'USED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-3.5 h-3.5" />
            USED
          </span>
        );

      case 'UNUSED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <AlertCircle className="w-3.5 h-3.5" />
            UNUSED
          </span>
        );

      case 'UNKNOWN':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <HelpCircle className="w-3.5 h-3.5" />
            UNKNOWN
          </span>
        );
    }
  };

  return (
    <div className="bg-surface-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
      <div className="p-4 border-b border-slate-800 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500" />

          <input
            type="text"
            placeholder="Search custom packages..."
            value={searchTerm}
            onChange={(event) =>
              setSearchTerm(event.target.value)
            }
            className="w-full bg-surface-950 border border-slate-800 rounded-lg pl-9 pr-4 py-2 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500"
          />
        </div>

        <div className="flex items-center space-x-2 w-full md:w-auto">
          {(['ALL', 'USED', 'UNUSED', 'UNKNOWN'] as const).map(
            (status) => (
              <button
                key={status}
                type="button"
                onClick={() => setStatusFilter(status)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                  statusFilter === status
                    ? 'bg-slate-700 text-white'
                    : 'bg-surface-950 text-slate-400 hover:text-slate-200 border border-slate-800'
                }`}
              >
                {status}
              </button>
            ),
          )}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-surface-950/50 border-b border-slate-800 text-slate-400 font-semibold">
              <th className="p-3.5">Status</th>
              <th className="p-3.5">Package Identifier</th>
              <th className="p-3.5 text-center">Detected Nodes</th>
              <th className="p-3.5 text-center">Referenced Nodes</th>
              <th className="p-3.5">Filesystem Location</th>
              <th className="p-3.5 text-right">Actions</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-800 text-slate-300">
            {filtered.length === 0 ? (
              <tr>
                <td
                  colSpan={6}
                  className="p-8 text-center text-slate-500"
                >
                  No custom node packages matched your query.
                </td>
              </tr>
            ) : (
              filtered.map((pkg) => (
                <tr
                  key={pkg.id}
                  className="hover:bg-slate-800/40 transition"
                >
                  <td className="p-3.5">
                    {getStatusBadge(pkg.status)}
                  </td>

                  <td className="p-3.5 font-medium text-slate-200">
                    {pkg.package_name}
                  </td>

                  <td className="p-3.5 text-center font-mono">
                    {pkg.detected_nodes_count}
                  </td>

                  <td className="p-3.5 text-center font-mono">
                    <span
                      className={
                        pkg.used_nodes_count > 0
                          ? 'text-emerald-400 font-bold'
                          : 'text-slate-500'
                      }
                    >
                      {pkg.used_nodes_count}
                    </span>
                  </td>

                  <td
                    className="p-3.5 font-mono text-[11px] text-slate-500 truncate max-w-xs"
                    title={pkg.directory_path}
                  >
                    {pkg.directory_path}
                  </td>

                  <td className="p-3.5 text-right">
                    <button
                      type="button"
                      onClick={() => onSelectPackage(pkg)}
                      className="p-1.5 hover:bg-slate-700 text-slate-400 hover:text-slate-200 rounded-md transition"
                      title="Inspect Nodes & Source AST"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
