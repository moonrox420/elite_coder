import React from 'react';
import { ShieldCheck } from 'lucide-react';

export const Navbar: React.FC = () => {
  return (
    <header className="bg-surface-900/80 backdrop-blur border-b border-slate-800 sticky top-0 z-20">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-brand-600 flex items-center justify-center">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>

          <div>
            <h1 className="text-sm font-bold text-slate-100 leading-tight">
              ComfyAudit
            </h1>
            <p className="text-[11px] text-slate-400 leading-tight">
              Enterprise Package Usage Auditor
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5" />
            Scanner Online
          </span>
        </div>
      </div>
    </header>
  );
};
