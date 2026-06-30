import React, { useState, useEffect } from 'react';
import { ChevronLeft, RefreshCw, Trophy, Swords, AlertTriangle } from 'lucide-react';

const CONTESTANT_COLORS = ['#00f2fe', '#bf55ec', '#10b981', '#f59e0b', '#ef4444'];

export default function ArenaDetails({ runId, onBack, setToast }) {
  const [run, setRun] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedCaseIdx, setSelectedCaseIdx] = useState(0);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/arena-runs/${runId}`)
      .then(r => { if (!r.ok) throw new Error('Failed to load arena run'); return r.json(); })
      .then(data => { setRun(data); setSelectedCaseIdx(0); })
      .catch(err => setToast({ type: 'error', message: err.message }))
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '300px' }}>
      <RefreshCw className="spinner text-primary" size={32} />
      <span style={{ marginLeft: '12px', fontSize: '14px', color: 'var(--text-secondary)' }}>Loading arena results…</span>
    </div>
  );

  if (!run) return (
    <div className="glass-card" style={{ padding: '40px', textAlign: 'center' }}>
      <AlertTriangle size={32} className="text-warning" style={{ marginBottom: '12px' }} />
      <button className="btn btn-secondary" style={{ marginTop: '16px' }} onClick={onBack}>Back</button>
    </div>
  );

  const leaderboard = run.leaderboard || [];
  const results = run.results || [];
  const selectedCase = results[selectedCaseIdx];

  const winner = leaderboard[0];
  const colorFor = (label) => {
    const idx = run.contestants.findIndex(c => c.label === label);
    return CONTESTANT_COLORS[idx >= 0 ? idx % CONTESTANT_COLORS.length : 0];
  };

  return (
    <div className="fade-in">
      {/* Breadcrumb */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
        <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: '13px', display: 'flex', alignItems: 'center' }} onClick={onBack}>
          <ChevronLeft size={16} /> Back to Arena History
        </button>
        <span style={{ color: 'var(--text-muted)' }}>/</span>
        <span style={{ fontWeight: 600, fontSize: '14px' }}>{run.name}</span>
      </div>

      <div style={{ marginBottom: '28px' }}>
        <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Swords size={28} style={{ color: 'var(--color-primary)' }} /> {run.name}
        </h1>
        <p className="page-subtitle">Dataset: <strong>{run.dataset_name}</strong> · {run.total_cases} cases · {run.contestants?.length} contestants</p>
      </div>

      {/* Leaderboard */}
      <div className="glass-card" style={{ marginBottom: '28px' }}>
        <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Trophy size={18} style={{ color: 'var(--color-warning)' }} /> Arena Leaderboard
        </h3>
        <div className="table-wrapper">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Contestant</th>
                <th>Provider</th>
                <th>Wins</th>
                <th>Ties</th>
                <th>Losses</th>
                <th>Win Rate</th>
                <th>Avg Correctness</th>
                <th>Avg Latency</th>
                <th>Total Cost</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((c, idx) => (
                <tr key={c.label}>
                  <td style={{ fontWeight: 'bold' }}>
                    {idx === 0 ? '🏆 1' : idx === 1 ? '🥈 2' : idx === 2 ? '🥉 3' : `${idx + 1}`}
                  </td>
                  <td>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: colorFor(c.label), flexShrink: 0 }} />
                      <strong>{c.label}</strong>
                    </span>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{c.model_name}</span>
                  </td>
                  <td><span className="badge badge-info">{c.model_provider?.toUpperCase()}</span></td>
                  <td style={{ color: 'var(--color-success)', fontWeight: 700 }}>{c.wins}</td>
                  <td style={{ color: 'var(--color-warning)' }}>{c.ties}</td>
                  <td style={{ color: 'var(--color-danger)' }}>{c.losses}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ flex: 1, height: '6px', background: 'var(--bg-input)', borderRadius: '3px', maxWidth: '80px' }}>
                        <div style={{ width: `${(c.win_rate * 100).toFixed(0)}%`, height: '100%', borderRadius: '3px', background: colorFor(c.label) }} />
                      </div>
                      <span style={{ fontWeight: 600, fontSize: '13px' }}>{(c.win_rate * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td><span className="rating-stars">{"★".repeat(Math.round(c.avg_correctness))}</span> <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>({c.avg_correctness})</span></td>
                  <td>{c.avg_latency}s</td>
                  <td>${c.total_cost.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary stat cards */}
      <div className="card-grid-4" style={{ marginBottom: '28px' }}>
        {leaderboard.slice(0, 4).map((c, idx) => (
          <div key={c.label} className="glass-card" style={{ padding: '16px 20px', borderColor: colorFor(c.label) + '55' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: colorFor(c.label) }} />
              <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>{c.label}</span>
            </div>
            <div style={{ display: 'flex', gap: '20px' }}>
              <div><div style={{ fontSize: '22px', fontWeight: 700, color: colorFor(c.label) }}>{c.wins}W</div><div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Wins</div></div>
              <div><div style={{ fontSize: '22px', fontWeight: 700 }}>{c.avg_correctness}</div><div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Correctness</div></div>
              <div><div style={{ fontSize: '22px', fontWeight: 700 }}>{c.avg_latency}s</div><div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Latency</div></div>
            </div>
          </div>
        ))}
      </div>

      {/* Per-case split view */}
      <div className="split-view">
        {/* Left: case list */}
        <div className="split-sidebar">
          {results.map((item, idx) => {
            const isTie = item.winner === 'tie';
            const winnerColor = isTie ? 'var(--color-warning)' : colorFor(item.winner);
            return (
              <div key={item.case_id}
                className={`case-item-card ${idx === selectedCaseIdx ? 'active' : ''}`}
                onClick={() => setSelectedCaseIdx(idx)}
              >
                <div style={{ minWidth: 0, flex: 1, marginRight: '8px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '2px' }}>CASE #{idx + 1}</div>
                  <div className="case-preview-text">{item.question}</div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '3px', flexShrink: 0 }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: winnerColor }} title={isTie ? 'Tie' : `Winner: ${item.winner}`} />
                  <span style={{ fontSize: '10px', color: winnerColor, fontWeight: 600 }}>{isTie ? 'TIE' : 'WIN'}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Right: case drilldown */}
        <div className="split-detail">
          {selectedCase && (
            <div className="glass-card">
              <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '16px', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h3 style={{ fontSize: '17px', fontWeight: 600, marginBottom: '6px' }}>Case {selectedCaseIdx + 1} — {selectedCase.case_id}</h3>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {selectedCase.winner === 'tie'
                      ? <span className="badge badge-warning">⚖ Tie</span>
                      : <span className="badge" style={{ background: colorFor(selectedCase.winner) + '22', color: colorFor(selectedCase.winner), border: `1px solid ${colorFor(selectedCase.winner)}44` }}>
                          🏆 {selectedCase.winner}
                        </span>
                    }
                  </div>
                </div>
              </div>

              {/* Question */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '8px' }}>Question</div>
                <div className="detail-text-box" style={{ fontSize: '14px' }}>{selectedCase.question}</div>
              </div>

              {/* Judge verdict */}
              {selectedCase.winner_reason && (
                <div style={{ marginBottom: '20px', padding: '12px 16px', background: 'rgba(0,242,254,0.03)', border: '1px solid rgba(0,242,254,0.15)', borderRadius: 'var(--radius-md)', fontSize: '13px', color: 'var(--text-secondary)' }}>
                  <strong style={{ color: 'var(--color-primary)', display: 'block', marginBottom: '4px' }}>Judge Reasoning</strong>
                  {selectedCase.winner_reason}
                </div>
              )}

              {/* Side-by-side responses */}
              <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '12px' }}>
                Contestant Responses
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(selectedCase.contestant_results?.length || 1, 2)}, 1fr)`, gap: '16px' }}>
                {selectedCase.contestant_results?.map((cr, i) => {
                  const isWinner = selectedCase.winner === cr.label;
                  const color = colorFor(cr.label);
                  return (
                    <div key={cr.label} style={{ border: `1px solid ${isWinner ? color : 'var(--border-color)'}`, borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
                      <div style={{ padding: '10px 14px', background: isWinner ? color + '18' : 'var(--bg-input)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, fontSize: '13px' }}>
                          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }} />
                          {cr.label}
                        </span>
                        {isWinner && selectedCase.winner !== 'tie' && <span style={{ fontSize: '11px', fontWeight: 700, color }}>🏆 WINNER</span>}
                      </div>
                      <div style={{ padding: '14px', fontSize: '13px', lineHeight: '1.7', color: 'var(--text-secondary)', maxHeight: '260px', overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                        {cr.model_answer || '(no response)'}
                      </div>
                      <div style={{ padding: '10px 14px', background: 'rgba(255,255,255,0.02)', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '16px', fontSize: '11px', color: 'var(--text-muted)' }}>
                        <span>✓ {cr.metrics?.llm_correctness}/5</span>
                        <span>≈ {cr.metrics?.llm_completeness}/5</span>
                        <span>💬 {cr.metrics?.llm_clarity}/5</span>
                        <span>⚡ {cr.metrics?.latency}s</span>
                        <span>~ {(cr.metrics?.similarity * 100).toFixed(0)}% sim</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Ideal answer */}
              <div style={{ marginTop: '20px' }}>
                <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '8px' }}>Golden Reference Answer</div>
                <div className="detail-text-box" style={{ fontSize: '13px', color: 'var(--color-success)', borderColor: 'rgba(16,185,129,0.2)', whiteSpace: 'pre-wrap' }}>
                  {selectedCase.ideal_answer}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
