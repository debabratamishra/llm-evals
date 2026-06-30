import React, { useState, useEffect } from 'react';
import { ChevronLeft, Filter, RefreshCw, ThumbsUp, ThumbsDown, Award, Clock, DollarSign, Brain, FileText, CheckCircle, AlertTriangle } from 'lucide-react';

export default function RunDetails({ runId, onBack, setToast }) {
  const [run, setRun] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedCaseIdx, setSelectedCaseIdx] = useState(0);
  const [filterMode, setFilterMode] = useState('all'); // 'all', 'failed', 'mismatch'
  const [activeTurnIdx, setActiveTurnIdx] = useState(0);

  useEffect(() => {
    setActiveTurnIdx(0);
  }, [selectedCaseIdx, filterMode]);

  const fetchRunDetails = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/runs/${runId}`);
      if (!res.ok) throw new Error('Failed to retrieve run details');
      const data = await res.json();
      setRun(data);
      setSelectedCaseIdx(0);
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRunDetails();
  }, [runId]);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '300px' }}>
        <RefreshCw className="spinner text-primary" size={32} />
        <span style={{ marginLeft: '12px', fontSize: '14px', color: 'var(--text-secondary)' }}>Loading run results...</span>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="glass-card" style={{ padding: '40px', textAlign: 'center' }}>
        <AlertTriangle size={32} className="text-warning" style={{ marginBottom: '12px' }} />
        <h3>Run Details Unavailable</h3>
        <button className="btn btn-secondary" style={{ marginTop: '16px' }} onClick={onBack}>
          Back to List
        </button>
      </div>
    );
  }

  // Filter cases based on selected filter mode
  const filteredCases = run.results.map((c, originalIdx) => ({ ...c, originalIdx })).filter(item => {
    if (filterMode === 'failed') {
      return item.metrics.llm_correctness < 3;
    }
    if (filterMode === 'mismatch') {
      return item.metrics.exact_match === 0;
    }
    return true;
  });

  const selectedCase = filteredCases.length > 0 && selectedCaseIdx < filteredCases.length
    ? filteredCases[selectedCaseIdx]
    : null;

  const activeTurn = selectedCase && selectedCase.is_multi_turn && selectedCase.turns
    ? selectedCase.turns[activeTurnIdx]
    : null;

  const displayData = selectedCase && selectedCase.is_multi_turn && activeTurn
    ? {
        question: activeTurn.user_message,
        ideal_answer: activeTurn.ideal_response,
        model_answer: activeTurn.model_response,
        metrics: activeTurn.metrics
      }
    : selectedCase;

  return (
    <div className="fade-in">
      {/* Header breadcrumb */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
        <button 
          className="btn btn-secondary" 
          style={{ padding: '6px 12px', fontSize: '13px', display: 'flex', alignItems: 'center' }} 
          onClick={onBack}
        >
          <ChevronLeft size={16} /> Back to Runs
        </button>
        <span style={{ color: 'var(--text-muted)' }}>/</span>
        <span style={{ fontWeight: 600, fontSize: '14px' }}>{run.name}</span>
      </div>

      <div className="header-container" style={{ marginBottom: '24px' }}>
        <div>
          <h1 className="page-title">{run.name}</h1>
          <p className="page-subtitle" style={{ marginBottom: '10px' }}>
            Model: <strong style={{ color: 'var(--color-primary)' }}>{run.model_name}</strong> ({run.model_provider}) | Dataset: <strong>{run.dataset_name}</strong>
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', fontSize: '11px', color: 'var(--text-secondary)', background: 'rgba(255,255,255,0.02)', padding: '6px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', width: 'fit-content' }}>
            <span><strong>Temp:</strong> {run.parameters?.temperature ?? 'default'}</span>
            {run.parameters?.max_tokens !== undefined && run.parameters?.max_tokens !== null && (
              <span>| <strong>Max Tokens:</strong> {run.parameters.max_tokens}</span>
            )}
            {run.parameters?.top_p !== undefined && run.parameters?.top_p !== null && (
              <span>| <strong>Top P:</strong> {run.parameters.top_p}</span>
            )}
            {run.parameters?.frequency_penalty !== undefined && run.parameters?.frequency_penalty !== null && (
              <span>| <strong>Freq Penalty:</strong> {run.parameters.frequency_penalty}</span>
            )}
            {run.parameters?.presence_penalty !== undefined && run.parameters?.presence_penalty !== null && (
              <span>| <strong>Pres Penalty:</strong> {run.parameters.presence_penalty}</span>
            )}
            {run.parameters?.system_prompt && (
              <span>| <strong>System:</strong> <span style={{ fontStyle: 'italic' }} title={run.parameters.system_prompt}>"{run.parameters.system_prompt.substring(0, 40)}{run.parameters.system_prompt.length > 40 ? '...' : ''}"</span></span>
            )}
            {run.parameters?.multi_turn_history_mode && (
              <span>| <strong>History Mode:</strong> {run.parameters.multi_turn_history_mode === 'model_response' ? '⚡ Model Response' : '📐 Teacher Forcing'}</span>
            )}
          </div>
        </div>
      </div>

      {/* Summary statistics row */}
      <div className="card-grid-4" style={{ marginBottom: '24px' }}>
        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div className="card-header" style={{ marginBottom: '6px' }}>
            <span className="card-title" style={{ fontSize: '11px' }}>Avg Correctness</span>
            <Award size={14} className="text-secondary" />
          </div>
          <div className="card-value" style={{ fontSize: '24px' }}>{run.metrics.avg_correctness} <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>/ 5</span></div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div className="card-header" style={{ marginBottom: '6px' }}>
            <span className="card-title" style={{ fontSize: '11px' }}>Avg Latency</span>
            <Clock size={14} className="text-secondary" />
          </div>
          <div className="card-value" style={{ fontSize: '24px' }}>{run.metrics.avg_latency}s</div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div className="card-header" style={{ marginBottom: '6px' }}>
            <span className="card-title" style={{ fontSize: '11px' }}>Average Similarity</span>
            <Brain size={14} className="text-secondary" />
          </div>
          <div className="card-value" style={{ fontSize: '24px' }}>{(run.metrics.avg_similarity * 100).toFixed(0)}%</div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div className="card-header" style={{ marginBottom: '6px' }}>
            <span className="card-title" style={{ fontSize: '11px' }}>Total Cost</span>
            <DollarSign size={14} className="text-secondary" />
          </div>
          <div className="card-value" style={{ fontSize: '24px' }}>${run.metrics.total_cost.toFixed(4)}</div>
        </div>
      </div>

      {/* Filters row */}
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '20px' }}>
        <span style={{ fontSize: '13px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Filter size={14} /> Filter cases:
        </span>
        <button 
          className={`btn btn-secondary ${filterMode === 'all' ? 'active' : ''}`}
          style={{ padding: '6px 14px', fontSize: '12px', background: filterMode === 'all' ? 'var(--bg-input)' : '', borderColor: filterMode === 'all' ? 'var(--color-primary)' : '' }}
          onClick={() => { setFilterMode('all'); setSelectedCaseIdx(0); }}
        >
          All Cases ({run.results.length})
        </button>
        <button 
          className={`btn btn-secondary ${filterMode === 'failed' ? 'active' : ''}`}
          style={{ padding: '6px 14px', fontSize: '12px', color: 'var(--color-danger)', background: filterMode === 'failed' ? 'var(--bg-input)' : '', borderColor: filterMode === 'failed' ? 'var(--color-danger)' : '' }}
          onClick={() => { setFilterMode('failed'); setSelectedCaseIdx(0); }}
        >
          Low Correctness (Score &lt; 3) ({run.results.filter(c => c.metrics.llm_correctness < 3).length})
        </button>
        <button 
          className={`btn btn-secondary ${filterMode === 'mismatch' ? 'active' : ''}`}
          style={{ padding: '6px 14px', fontSize: '12px', color: 'var(--color-warning)', background: filterMode === 'mismatch' ? 'var(--bg-input)' : '', borderColor: filterMode === 'mismatch' ? 'var(--color-warning)' : '' }}
          onClick={() => { setFilterMode('mismatch'); setSelectedCaseIdx(0); }}
        >
          Mismatches (Exact Match = 0) ({run.results.filter(c => c.metrics.exact_match === 0).length})
        </button>
      </div>

      {/* Split view */}
      <div className="split-view">
        {/* Left Side: Cases List */}
        <div className="split-sidebar">
          {filteredCases.length === 0 ? (
            <div className="glass-card" style={{ padding: '24px', textAlign: 'center', fontSize: '13px', color: 'var(--text-muted)' }}>
              No cases match the selected filter.
            </div>
          ) : (
            filteredCases.map((item, idx) => {
              const isCorrect = item.metrics.llm_correctness >= 4;
              const isAverage = item.metrics.llm_correctness === 3;
              const isError = item.metrics.llm_correctness < 3;
              
              const statusClass = isCorrect ? 'badge-success' : isAverage ? 'badge-warning' : 'badge-danger';
              const statusText = isCorrect ? 'PASS' : isAverage ? 'WARN' : 'FAIL';

              return (
                <div 
                  key={item.case_id}
                  className={`case-item-card ${idx === selectedCaseIdx ? 'active' : ''}`}
                  onClick={() => setSelectedCaseIdx(idx)}
                >
                  <div style={{ minWidth: 0, marginRight: '8px', flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                      <span style={{ fontWeight: 600, fontSize: '12px', color: 'var(--text-muted)' }}>
                        CASE #{item.originalIdx + 1}
                      </span>
                      {item.is_multi_turn && (
                        <span className="badge badge-info" style={{ fontSize: '9px', padding: '1px 4px' }}>
                          {item.turns?.length} Turns
                        </span>
                      )}
                    </div>
                    <div className="case-preview-text">{item.question}</div>
                  </div>
                  <span className={`badge ${statusClass} case-score-badge`}>
                    {item.metrics.llm_correctness} ★
                  </span>
                </div>
              );
            })
          )}
        </div>

        {/* Right Side: Case Drilldown Detail */}
        <div className="split-detail">
          {selectedCase ? (
            <div className="glass-card">
              <div className="card-header" style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '16px', marginBottom: '20px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: 600 }}>Case Details (ID: {selectedCase.case_id})</h3>
                <span className={`badge ${
                  selectedCase.metrics.llm_correctness >= 4 ? 'badge-success' : selectedCase.metrics.llm_correctness === 3 ? 'badge-warning' : 'badge-danger'
                }`}>
                  {selectedCase.is_multi_turn ? `Case Avg: ${selectedCase.metrics.llm_correctness}/5` : `LLM Score: ${selectedCase.metrics.llm_correctness}/5`}
                </span>
              </div>

              {selectedCase.is_multi_turn && (
                <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', overflowX: 'auto' }}>
                  {selectedCase.turns.map((turn, tIdx) => (
                    <button
                      key={tIdx}
                      className={`btn ${activeTurnIdx === tIdx ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '6px 12px', fontSize: '12px', minWidth: '80px' }}
                      type="button"
                      onClick={() => setActiveTurnIdx(tIdx)}
                    >
                      Turn {tIdx + 1}
                    </button>
                  ))}
                </div>
              )}

              {/* Performance Score Cards grid */}
              <div className="score-grid">
                <div className="score-tile">
                  <div className="score-tile-label">Correctness</div>
                  <div className="score-tile-val" style={{ color: displayData.metrics.llm_correctness >= 4 ? 'var(--color-success)' : displayData.metrics.llm_correctness === 3 ? 'var(--color-warning)' : 'var(--color-danger)' }}>
                    {displayData.metrics.llm_correctness}/5
                  </div>
                </div>

                <div className="score-tile">
                  <div className="score-tile-label">Completeness</div>
                  <div className="score-tile-val">{displayData.metrics.llm_completeness}/5</div>
                </div>

                <div className="score-tile">
                  <div className="score-tile-label">Clarity</div>
                  <div className="score-tile-val">{displayData.metrics.llm_clarity}/5</div>
                </div>

                <div className="score-tile">
                  <div className="score-tile-label">Similarity</div>
                  <div className="score-tile-val">{(displayData.metrics.similarity * 100).toFixed(0)}%</div>
                </div>

                <div className="score-tile">
                  <div className="score-tile-label">Exact Match</div>
                  <div className="score-tile-val">{displayData.metrics.exact_match === 1 ? 'YES' : 'NO'}</div>
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <div className="detail-section-header">
                  {selectedCase.is_multi_turn ? `User Prompt (Turn ${activeTurnIdx + 1})` : 'Question'}
                </div>
                <div className="detail-text-box" style={{ fontFamily: 'var(--font-sans)', fontSize: '15px' }}>{displayData.question}</div>
              </div>

              {/* Side-by-side comparative answers output */}
              <div className="side-by-side" style={{ marginBottom: '20px' }}>
                <div>
                  <div className="detail-section-header">Golden Reference Answer</div>
                  <div className="detail-text-box" style={{ borderLeft: '3px solid var(--color-success)', background: 'rgba(16, 185, 129, 0.015)' }}>{displayData.ideal_answer}</div>
                </div>

                <div>
                  <div className="detail-section-header">Model Generated Answer</div>
                  <div className={`detail-text-box ${displayData.metrics.llm_correctness < 3 ? 'mismatched' : ''}`} style={{ 
                    borderLeft: displayData.metrics.llm_correctness >= 4 ? '3px solid var(--color-primary)' : displayData.metrics.llm_correctness === 3 ? '3px solid var(--color-warning)' : '3px solid var(--color-danger)', 
                    background: 'rgba(0, 242, 254, 0.015)' 
                  }}>
                    {displayData.model_answer}
                  </div>
                </div>
              </div>

              {/* Evaluation Rationale — Rubric Score Cards */}
              <div>
                <div className="detail-section-header">Judge Evaluation &amp; Rubric Scores</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '12px' }}>
                  {[
                    { label: 'Correctness', key: 'llm_correctness', desc: 'Factual accuracy vs golden answer' },
                    { label: 'Completeness', key: 'llm_completeness', desc: 'Coverage of all expected points' },
                    { label: 'Clarity', key: 'llm_clarity', desc: 'Structure, coherence & formatting' },
                  ].map(({ label, key, desc }) => {
                    const score = displayData.metrics[key] ?? 0;
                    const isFloat = !Number.isInteger(score);
                    const displayScore = isFloat ? score.toFixed(1) : score;
                    const pct = Math.round((score / 5) * 100);
                    const color = score >= 4 ? 'var(--color-success)' : score === 3 ? 'var(--color-warning)' : 'var(--color-danger)';
                    return (
                      <div key={key} style={{
                        padding: '14px',
                        borderRadius: 'var(--radius-md)',
                        border: `1px solid ${color}44`,
                        background: `${color}0a`,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>{label}</span>
                          <span style={{ fontSize: '20px', fontWeight: 700, color }}>{displayScore}<span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 400 }}>/5</span></span>
                        </div>
                        <div style={{ height: '4px', borderRadius: '2px', background: 'rgba(255,255,255,0.08)' }}>
                          <div style={{ height: '100%', width: `${pct}%`, borderRadius: '2px', background: color, transition: 'width 0.4s ease' }} />
                        </div>
                        <p style={{ fontSize: '10px', color: 'var(--text-muted)', margin: 0 }}>{desc}</p>
                      </div>
                    );
                  })}
                </div>
                <div className="detail-text-box" style={{ fontSize: '13px', fontStyle: 'italic', color: 'var(--text-secondary)', borderLeft: '3px solid var(--color-primary)', lineHeight: '1.65' }}>
                  <Brain size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px', opacity: 0.6 }} />
                  {displayData.metrics.reason}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border-color)', paddingTop: '16px', marginTop: '24px', fontSize: '12px', color: 'var(--text-muted)' }}>
                <span>Latency: <strong>{displayData.metrics.latency}s</strong></span>
                <span>Token Usage: Input: <strong>{displayData.metrics.input_tokens}</strong> | Output: <strong>{displayData.metrics.output_tokens}</strong></span>
                <span>Estimated Cost: <strong>${displayData.metrics.cost.toFixed(6)}</strong></span>
              </div>
            </div>
          ) : (
            <div className="glass-card" style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
              Select a test case from the sidebar to inspect detailed output comparisons and metrics.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
