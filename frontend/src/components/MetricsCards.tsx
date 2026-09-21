import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Package,
  Search,
} from 'lucide-react';
import { AuditScanDetail } from '../types';

interface MetricsCardsProps {
  scan: AuditScanDetail | null;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({ scan }) => {
  const packages = scan?.packages ?? [];
  const usedCount = packages.filter(
    (pkg) => pkg.status === 'USED',
  ).length;
  const unusedCount = packages.filter(
    (pkg) => pkg.status === 'UNUSED',
  ).length;
  const unknownCount = packages.filter(
    (pkg) => pkg.status === 'UNKNOWN',
  ).length;
  const totalNodes = packages.reduce(
    (sum, pkg) => sum + pkg.detected_nodes_count,
    0,
  );

  const cards = [
    {
      label: 'Total Custom Packages',
      value: packages.length,
      sub: scan
        ? scan.scanned_root_path === 'AUTO_DISCOVERY'
          ? 'Auto-discovery'
          : scan.scanned_root_path
        : 'No scan loaded',
      icon: Package,
      accent: 'text-brand-400',
    },
    {
      label: 'Packages In Use',
      value: usedCount,
      sub: 'Referenced by workflows',
      icon: CheckCircle2,
      accent: 'text-emerald-400',
    },
    {
      label: 'Potentially Unused',
      value: unusedCount,
      sub: 'Not referenced anywhere',
      icon: AlertTriangle,
      accent: 'text-amber-400',
    },
    {
      label: 'Detected Nodes',
      value: totalNodes,
      sub: `${unknownCount} unresolved package${unknownCount === 1 ? '' : 's'}`,
      icon: Search,
      accent: 'text-purple-400',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => (
        <div
          key={card.label}
          className="bg-surface-900 border border-slate-800 rounded-xl p-4 shadow-lg"
        >
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide">
              {card.label}
            </p>
            <card.icon className={`w-4 h-4 ${card.accent}`} />
          </div>

          <p className="mt-2 text-2xl font-bold text-slate-100">
            {card.value}
          </p>

          <p
            className="mt-1 text-[11px] text-slate-500 truncate"
            title={card.sub}
          >
            {card.sub}
          </p>
        </div>
      ))}
    </div>
  );
};
