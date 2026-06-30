import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis, Label } from 'recharts';
import { Award, Zap, DollarSign, Database, Activity, TrendingUp } from 'lucide-react';

export default function DashboardOverview({ runs, onViewRun }) {
  // Aggregate data
  const totalRuns = runs.length;
  const uniqueModels = new Set(runs.map(r => `${r.model_provider}/${r.model_name}`)).size;
  
  const avgCorrectness = runs.length 
    ? (runs.reduce((acc, r) => acc + (r.metrics.avg_correctness || 0), 0) / runs.length).toFixed(2)
    : '0.00';
    
  const totalCost = runs.length
    ? runs.reduce((acc, r) => acc + (r.metrics.total_cost || 0), 0).toFixed(4)
    : '0.0000';

  // Format model comparison charts data
  // Combine multiple runs of the same model and average them, or take the latest
  const modelMap = {};
  runs.forEach(run => {
    const key = `${run.model_provider}/${run.model_name}`;
    if (!modelMap[key] || new Date(run.created_at) > new Date(modelMap[key].created_at)) {
      modelMap[key] = run;
    }
  });

  const chartData = Object.values(modelMap).map(run => {
    const formattedName = run.model_name
      .replace('meta-llama/', '')
      .replace('microsoft/', '')
      .replace('-instruct', '')
      .replace('-Instruct', '');
      
    return {
      name: formattedName,
      correctness: run.metrics.avg_correctness || 0,
      completeness: run.metrics.avg_completeness || 0,
      clarity: run.metrics.avg_clarity || 0,
      latency: run.metrics.avg_latency || 0,
      cost: run.metrics.total_cost || 0,
      accuracy: run.metrics.avg_similarity * 100 || 0,
      provider: run.model_provider,
      runId: run.id
    };
  });

  // Calculate efficiency leaderboard
  const leaderboard = [...chartData]
    .map(model => {
      // Custom efficiency score: Correctness / (Cost * 100 + 1)
      const costFactor = model.cost * 1000 + 1;
      const score = (model.correctness / costFactor).toFixed(2);
      return { ...model, efficiencyScore: parseFloat(score) };
    })
    .sort((a, b) => b.efficiencyScore - a.efficiencyScore);

  return (
    <div className="fade-in">
      {/* Metrics Row */}
      <div className="card-grid-4">
        <div className="glass-card accented">
          <div className="card-header">
            <span className="card-title">Total Runs</span>
            <div className="card-icon-wrapper">
              <Activity size={18} />
            </div>
          </div>
          <div className="card-value">{totalRuns}</div>
          <div className="card-desc">Completed evaluations</div>
        </div>

        <div className="glass-card accented">
          <div className="card-header">
            <span className="card-title">Models Tested</span>
            <div className="card-icon-wrapper">
              <Database size={18} />
            </div>
          </div>
          <div className="card-value">{uniqueModels}</div>
          <div className="card-desc">Unique model configurations</div>
        </div>

        <div className="glass-card accented">
          <div className="card-header">
            <span className="card-title">Avg Correctness</span>
            <div className="card-icon-wrapper">
              <Award size={18} />
            </div>
          </div>
          <div className="card-value">{avgCorrectness} <span style={{ fontSize: '16px', color: 'var(--text-secondary)' }}>/ 5</span></div>
          <div className="card-desc">LLM-as-a-judge average grade</div>
        </div>

        <div className="glass-card accented">
          <div className="card-header">
            <span className="card-title">Cumulative Cost</span>
            <div className="card-icon-wrapper">
              <DollarSign size={18} />
            </div>
          </div>
          <div className="card-value">${totalCost}</div>
          <div className="card-desc">Estimated API expense (USD)</div>
        </div>
      </div>

      {runs.length === 0 ? (
        <div className="glass-card" style={{ padding: '40px', textAlign: 'center' }}>
          <TrendingUp size={48} className="upload-icon" style={{ strokeWidth: 1 }} />
          <h3 style={{ marginBottom: '8px', fontSize: '18px' }}>No Evaluation Data Found</h3>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '500px', margin: '0 auto 20px' }}>
            To get started, go to the <strong>Run Eval</strong> tab to evaluate a model or upload test datasets. Seeding is automatically enabled with medical and reasoning benchmarks.
          </p>
        </div>
      ) : (
        <>
          {/* Charts Grid */}
          <div className="dashboard-grid">
            {/* Bar chart - Performance */}
            <div className="glass-card">
              <h3 style={{ marginBottom: '4px', fontSize: '16px', fontWeight: 600 }}>LLM Judge Scores</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '16px' }}>Detailed criteria scores out of 5 graded by LLM evaluator</p>
              
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} />
                    <YAxis domain={[0, 5]} stroke="var(--text-muted)" fontSize={11} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--bg-sidebar)', borderColor: 'var(--border-color)' }}
                      labelStyle={{ color: 'var(--text-primary)', fontWeight: 'bold' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Bar dataKey="correctness" name="Correctness" fill="#00f2fe" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="completeness" name="Completeness" fill="#4facfe" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="clarity" name="Clarity" fill="#bf55ec" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Scatter chart - Efficiency */}
            <div className="glass-card">
              <h3 style={{ marginBottom: '4px', fontSize: '16px', fontWeight: 600 }}>Benchmarking Efficiency Matrix</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '16px' }}>Similarity Score (%) vs Execution Latency (seconds)</p>
              
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis type="number" dataKey="latency" name="Latency" unit="s" stroke="var(--text-muted)" fontSize={11}>
                      <Label value="Avg Latency (s)" offset={-5} position="insideBottom" fill="var(--text-muted)" fontSize={11} />
                    </XAxis>
                    <YAxis type="number" dataKey="accuracy" name="Similarity" unit="%" stroke="var(--text-muted)" fontSize={11}>
                      <Label value="Similarity Score (%)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} fill="var(--text-muted)" fontSize={11} />
                    </YAxis>
                    <ZAxis type="number" dataKey="correctness" range={[60, 400]} />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      contentStyle={{ backgroundColor: 'var(--bg-sidebar)', borderColor: 'var(--border-color)' }}
                      formatter={(value, name) => [value, name]}
                    />
                    <Scatter name="Models" data={chartData} fill="#00f2fe">
                      {chartData.map((entry, index) => (
                        <circle
                          key={`circle-${index}`}
                          cx={0}
                          cy={0}
                          r={10}
                          fill={entry.provider === 'gemini' ? '#00f2fe' : entry.provider === 'openai' ? '#bf55ec' : '#10b981'}
                          style={{ cursor: 'pointer' }}
                          onClick={() => onViewRun(entry.runId)}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Model Leaderboard */}
          <div className="glass-card">
            <h3 style={{ marginBottom: '16px', fontSize: '16px', fontWeight: 600 }}>Leaderboard Rankings</h3>
            <div className="table-wrapper">
              <table className="custom-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model Configuration</th>
                    <th>Provider</th>
                    <th>Correctness Score</th>
                    <th>Avg Latency</th>
                    <th>Avg Cost (1M)</th>
                    <th>Efficiency Index</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((model, idx) => (
                    <tr key={idx}>
                      <td style={{ fontWeight: 'bold', width: '60px' }}>
                        {idx === 0 ? '🏆 1' : idx === 1 ? '🥈 2' : idx === 2 ? '🥉 3' : `${idx + 1}`}
                      </td>
                      <td style={{ fontWeight: 500 }}>{model.name}</td>
                      <td>
                        <span className={`badge ${
                          model.provider === 'gemini' ? 'badge-info' : model.provider === 'openai' ? 'badge-success' : 'badge-warning'
                        }`}>
                          {model.provider.toUpperCase()}
                        </span>
                      </td>
                      <td>
                        <span className="rating-stars">{"★".repeat(Math.round(model.correctness))}</span>
                        <span style={{ color: 'var(--text-muted)', fontSize: '12px', marginLeft: '6px' }}>({model.correctness.toFixed(2)})</span>
                      </td>
                      <td>{model.latency.toFixed(2)}s</td>
                      <td>${(model.cost * 1000).toFixed(4)}</td>
                      <td style={{ fontWeight: 'bold', color: 'var(--color-primary)' }}>{model.efficiencyScore}</td>
                      <td>
                        <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: '12px' }} onClick={() => onViewRun(model.runId)}>
                          View Run
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
