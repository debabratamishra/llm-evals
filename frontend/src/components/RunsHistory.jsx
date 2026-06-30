import React, { useState } from 'react';
import { Eye, Trash2, Calendar, FileText, BarChart } from 'lucide-react';

export default function RunsHistory({ runs, onViewRun, onRefresh, setToast }) {
  const [searchTerm, setSearchTerm] = useState('');

  const handleDelete = async (id, name, e) => {
    e.stopPropagation(); // Prevent row click
    if (!window.confirm(`Are you sure you want to delete the evaluation run "${name}"?`)) return;

    try {
      const res = await fetch(`/api/runs/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete run');
      
      setToast({ type: 'success', message: `Evaluation run "${name}" deleted` });
      onRefresh();
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    }
  };

  const filteredRuns = runs.filter(run => {
    const searchStr = `${run.name} ${run.model_name} ${run.dataset_name}`.toLowerCase();
    return searchStr.includes(searchTerm.toLowerCase());
  });

  return (
    <div className="fade-in">
      <div className="header-container">
        <div>
          <h1 className="page-title">Evaluation Runs</h1>
          <p className="page-subtitle">View logs, overall metrics, and comparative histories of completed evaluations.</p>
        </div>
        <div>
          <input 
            type="text" 
            className="form-control"
            style={{ width: '260px' }}
            placeholder="Search runs, models, datasets..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {filteredRuns.length === 0 ? (
        <div className="glass-card" style={{ padding: '60px', textAlign: 'center' }}>
          <BarChart size={48} className="upload-icon" style={{ strokeWidth: 1 }} />
          <h3 style={{ marginBottom: '8px', fontSize: '18px' }}>No Runs Recorded</h3>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '400px', margin: '0 auto' }}>
            {searchTerm ? 'No runs match your search query.' : 'No evaluation history was found. Run your first benchmark evaluation to populate this view.'}
          </p>
        </div>
      ) : (
        <div className="table-wrapper">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Run Name</th>
                <th>Model</th>
                <th>Dataset</th>
                <th>Average Correctness</th>
                <th>Avg Latency</th>
                <th>Total Cost</th>
                <th>Executed At</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map((run) => {
                const formattedModel = run.model_name
                  .replace('meta-llama/', '')
                  .replace('microsoft/', '')
                  .replace('-instruct', '')
                  .replace('-Instruct', '');
                
                const correctnessVal = run.metrics.avg_correctness || 0;
                
                return (
                  <tr 
                    key={run.id} 
                    onClick={() => onViewRun(run.id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{run.name}</div>
                    </td>
                    <td>
                      <span className={`badge ${
                        run.model_provider === 'gemini' ? 'badge-info' : run.model_provider === 'openai' ? 'badge-success' : 'badge-warning'
                      }`} style={{ marginRight: '6px' }}>
                        {run.model_provider.toUpperCase()}
                      </span>
                      {formattedModel}
                    </td>
                    <td>
                      <div style={{ color: 'var(--text-secondary)' }}>{run.dataset_name}</div>
                    </td>
                    <td>
                      <span className="rating-stars">{"★".repeat(Math.round(correctnessVal))}</span>
                      <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginLeft: '6px' }}>
                        ({correctnessVal.toFixed(2)})
                      </span>
                    </td>
                    <td>{run.metrics.avg_latency?.toFixed(2)}s</td>
                    <td>${run.metrics.total_cost?.toFixed(4)}</td>
                    <td style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      {new Date(run.created_at).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '8px' }}>
                        <button 
                          className="btn btn-secondary" 
                          style={{ padding: '8px', fontSize: '13px' }}
                          onClick={(e) => { e.stopPropagation(); onViewRun(run.id); }}
                        >
                          <Eye size={14} />
                        </button>
                        <button 
                          className="btn btn-secondary" 
                          style={{ padding: '8px', fontSize: '13px', color: 'var(--color-danger)' }}
                          onClick={(e) => handleDelete(run.id, run.name, e)}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
