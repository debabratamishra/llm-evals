import React from 'react';
import { Swords, Trophy, Trash2, Eye, RefreshCw, AlertTriangle } from 'lucide-react';

const CONTESTANT_COLORS = ['#00f2fe', '#bf55ec', '#10b981', '#f59e0b', '#ef4444'];

export default function ArenaHistory({ arenaRuns, onViewRun, onRefresh, setToast }) {
  const handleDelete = async (runId, runName) => {
    if (!window.confirm(`Delete arena run "${runName}"? This cannot be undone.`)) return;
    try {
      const res = await fetch(`/api/arena-runs/${runId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete arena run');
      setToast({ type: 'success', message: 'Arena run deleted.' });
      onRefresh();
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    }
  };

  if (!arenaRuns || arenaRuns.length === 0) {
    return (
      <div className="fade-in">
        <div className="header-container">
          <div>
            <h1 className="page-title">Arena History</h1>
            <p className="page-subtitle">Past head-to-head multi-model evaluations.</p>
          </div>
          <button className="btn btn-secondary" onClick={onRefresh} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <RefreshCw size={16} /> Refresh
          </button>
        </div>
        <div className="glass-card" style={{ padding: '48px', textAlign: 'center' }}>
          <Swords size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px', strokeWidth: 1 }} />
          <h3 style={{ marginBottom: '8px' }}>No Arena Runs Yet</h3>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '400px', margin: '0 auto' }}>
            Head to <strong>Arena Eval</strong> to pit multiple models against each other on the same dataset.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="fade-in">
      <div className="header-container">
        <div>
          <h1 className="page-title">Arena History</h1>
          <p className="page-subtitle">{arenaRuns.length} arena run{arenaRuns.length !== 1 ? 's' : ''} completed.</p>
        </div>
        <button className="btn btn-secondary" onClick={onRefresh} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <RefreshCw size={16} /> Refresh
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {arenaRuns.map((run) => {
          const leaderboard = run.leaderboard || [];
          const top = leaderboard[0];
          const ts = run.created_at ? new Date(run.created_at).toLocaleString() : '—';

          return (
            <div key={run.id} className="glass-card" style={{ padding: '20px 24px', cursor: 'pointer' }} onClick={() => onViewRun(run.id)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1, minWidth: 0, marginRight: '16px' }}>
                  {/* Title row */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                    <Swords size={18} style={{ color: 'var(--color-primary)', flexShrink: 0 }} />
                    <h3 style={{ fontSize: '16px', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{run.name}</h3>
                    {top && (
                      <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', fontWeight: 600, color: CONTESTANT_COLORS[0], flexShrink: 0 }}>
                        🏆 {top.label}
                      </span>
                    )}
                  </div>

                  {/* Meta row */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                    <span>📅 {ts}</span>
                    <span>📂 {run.dataset_name}</span>
                    <span>🎯 {run.total_cases} cases</span>
                  </div>

                  {/* Contestant pills */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {(run.contestants || []).map((c, idx) => {
                      const lb = leaderboard.find(l => l.label === c.label);
                      return (
                        <div key={c.label} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '4px 10px', background: 'var(--bg-input)', borderRadius: '20px', border: `1px solid ${CONTESTANT_COLORS[idx % CONTESTANT_COLORS.length]}44`, fontSize: '12px' }}>
                          <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: CONTESTANT_COLORS[idx % CONTESTANT_COLORS.length], flexShrink: 0 }} />
                          <span style={{ fontWeight: 500 }}>{c.label}</span>
                          {lb && (
                            <span style={{ color: 'var(--text-muted)' }}>
                              {lb.wins}W/{lb.ties}T/{lb.losses}L · {lb.avg_correctness}/5
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flexShrink: 0 }} onClick={e => e.stopPropagation()}>
                  <button className="btn btn-secondary" style={{ padding: '7px 14px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}
                    onClick={() => onViewRun(run.id)}>
                    <Eye size={14} /> View
                  </button>
                  <button className="btn btn-secondary" style={{ padding: '7px 14px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-danger)', borderColor: 'rgba(239,68,68,0.3)' }}
                    onClick={() => handleDelete(run.id, run.name)}>
                    <Trash2 size={14} /> Delete
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
