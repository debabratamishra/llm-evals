import React, { useState, useEffect } from 'react';
import { ChevronLeft, Filter, RefreshCw, ThumbsUp, ThumbsDown, Award, Clock, DollarSign, Brain, FileText, CheckCircle, AlertTriangle } from 'lucide-react';

export default function RunDetails({ runId, onBack, setToast }) {
  const [run, setRun] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedCaseIdx, setSelectedCaseIdx] = useState(0);
  const [filterMode, setFilterMode] = useState('all'); // 'all', 'failed', 'mismatch'

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
          <p className="page-subtitle">
            Model: <strong style={{ color: 'var(--color-primary)' }}>{run.model_name}</strong> ({run.model_provider}) | Dataset: <strong>{run.dataset_name}</strong>
          </p>
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
                  <div style={{ minWidth: 0, marginRight: '8px' }}>
                    <div style={{ fontWeight: 600, fontSize: '12px', color: 'var(--text-muted)', marginBottom: '2px' }}>
                      CASE #{item.originalIdx + 1}
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
                  LLM Score: {selectedCase.metrics.llm_correctness}/5
                </span>
              </div>

              {/* Performance Score Cards grid */}
              <div className="score-grid">
                <div className="score-tile">
                  <div className="score-tile-label">Correctness</div>
                  <div className="score-tile-val" style={{ color: selectedCase.metrics.llm_correctness >= 4 ? 'var(--color-success)' : selectedCase.metrics.llm_correctness === 3 ? 'var(--color-warning)' : 'var(--color-danger)' }}>
                    {selectedCase.metrics.llm_correctness}/5
                  </div>
                </div>

                <div className="score-tile">
                  <div className="score-tile-label">Completeness</div>
                  <div className="score-tile-val">{selectedCase.metrics.llm_completeness}/5</div>
                </div>

                <div className="score-tile">
                  <div className="score-tile-label">Clarity</div>
                  <div className="score-tile-val">{selectedCase.metrics.llm_clarity}/5</div>
                </div>

                <div className="score-tile">
                  <div className="score-tile-label">Similarity</div>
                  <div className="score-tile-val">{(selectedCase.metrics.similarity * 100).toFixed(0)}%</div>
                </div>

                <div className="score-tile">
                  <div className="score-tile-label">Exact Match</div>
                  <div className="score-tile-val">{selectedCase.metrics.exact_match === 1 ? 'YES' : 'NO'}</div>
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <div className="detail-section-header">Question</div>
                <div className="detail-text-box" style={{ fontFamily: 'var(--font-sans)', fontSize: '15px' }}>{selectedCase.question}</div>
              </div>

              {/* Side-by-side comparative answers output */}
              <div className="side-by-side" style={{ marginBottom: '20px' }}>
                <div>
                  <div className="detail-section-header">Golden Reference Answer</div>
                  <div className="detail-text-box" style={{ borderLeft: '3px solid var(--color-success)', background: 'rgba(16, 185, 129, 0.015)' }}>{selectedCase.ideal_answer}</div>
                </div>

                <div>
                  <div className="detail-section-header">Model Generated Answer</div>
                  <div className={`detail-text-box ${selectedCase.metrics.llm_correctness < 3 ? 'mismatched' : ''}`} style={{ 
                    borderLeft: selectedCase.metrics.llm_correctness >= 4 ? '3px solid var(--color-primary)' : selectedCase.metrics.llm_correctness === 3 ? '3px solid var(--color-warning)' : '3px solid var(--color-danger)', 
                    background: 'rgba(0, 242, 254, 0.015)' 
                  }}>
                    {selectedCase.model_answer}
                  </div>
                </div>
              </div>

              <div>
                <div className="detail-section-header">Evaluation Rationale</div>
                <div className="detail-text-box" style={{ fontSize: '13.5px', fontStyle: 'italic', color: 'var(--text-secondary)' }}>
                  {selectedCase.metrics.reason}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border-color)', paddingTop: '16px', marginTop: '24px', fontSize: '12px', color: 'var(--text-muted)' }}>
                <span>Latency: <strong>{selectedCase.metrics.latency}s</strong></span>
                <span>Token Usage: Input: <strong>{selectedCase.metrics.input_tokens}</strong> | Output: <strong>{selectedCase.metrics.output_tokens}</strong></span>
                <span>Estimated Cost: <strong>${selectedCase.metrics.cost.toFixed(6)}</strong></span>
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
