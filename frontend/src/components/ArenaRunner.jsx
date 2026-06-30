import React, { useState, useEffect } from 'react';
import { Swords, Plus, Trash2, Settings, Key, ShieldCheck, Loader2, Server, ExternalLink, AlertTriangle } from 'lucide-react';

// Default blank contestant template
const blankContestant = (idx) => ({
  _id: Date.now() + idx,
  label: '',
  model_provider: 'mock',
  model_name: 'llama-3.2-3b-mock',
  nvidiaNimModel: 'meta/llama-3.1-8b-instruct',
  openrouterModel: '',
  temperature: 0.2,
  system_prompt: '',
  max_tokens: '',
  top_p: '',
  frequency_penalty: '',
  presence_penalty: '',
  multi_turn_history_mode: 'model_response',
});

export default function ArenaRunner({ datasets, apiKeysSet, onArenaComplete, setToast }) {
  const [runName, setRunName] = useState('');
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [contestants, setContestants] = useState([blankContestant(0), blankContestant(1)]);

  // Shared credentials
  const [nvidiaNimBaseUrl, setNvidiaNimBaseUrl] = useState('https://integrate.api.nvidia.com/v1');
  const [nvidiaNimKey, setNvidiaNimKey] = useState('');
  const [openrouterKey, setOpenrouterKey] = useState('');

  const [isRunning, setIsRunning] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);

  const loadingSteps = [
    'Spinning up arena environment…',
    'Running contestant models in parallel…',
    'Receiving and recording all responses…',
    'Running LLM-as-a-Judge on each contestant…',
    'Running pairwise head-to-head comparisons…',
    'Tallying wins, losses, and ties…',
    'Building arena leaderboard…',
  ];

  useEffect(() => {
    if (apiKeysSet.nvidia_nim_base_url && apiKeysSet.nvidia_nim_base_url !== 'https://integrate.api.nvidia.com/v1') {
      setNvidiaNimBaseUrl(apiKeysSet.nvidia_nim_base_url);
    }
  }, [apiKeysSet.nvidia_nim_base_url]);

  useEffect(() => {
    if (datasets.length > 0 && !selectedDatasetId) setSelectedDatasetId(datasets[0].id);
  }, [datasets, selectedDatasetId]);

  useEffect(() => {
    if (!selectedDatasetId) return;
    const ds = datasets.find(d => d.id === selectedDatasetId);
    const dsName = ds ? ds.name : 'Dataset';
    const labels = contestants.map(c => c.label || getEffectiveModel(c) || 'Model').join(' vs ');
    setRunName(`Arena: ${labels} on ${dsName}`);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetId, contestants, datasets]);

  useEffect(() => {
    let interval;
    if (isRunning) {
      setLoadingStep(0);
      interval = setInterval(() => setLoadingStep(p => (p + 1) % loadingSteps.length), 2500);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  function getEffectiveModel(c) {
    if (c.model_provider === 'nvidia_nim') return c.nvidiaNimModel;
    if (c.model_provider === 'openrouter') return c.openrouterModel;
    return c.model_name;
  }

  function updateContestant(id, field, value) {
    setContestants(prev => prev.map(c => {
      if (c._id !== id) return c;
      const updated = { ...c, [field]: value };
      // Reset model name when provider changes
      if (field === 'model_provider') {
        if (value === 'mock') updated.model_name = 'llama-3.2-3b-mock';
        else if (value === 'nvidia_nim') updated.model_name = updated.nvidiaNimModel;
        else updated.model_name = updated.openrouterModel;
      }
      if (field === 'nvidiaNimModel') updated.model_name = value;
      if (field === 'openrouterModel') updated.model_name = value;
      return updated;
    }));
  }

  function addContestant() {
    if (contestants.length >= 5) {
      setToast({ type: 'error', message: 'Maximum 5 contestants allowed per arena run.' });
      return;
    }
    setContestants(prev => [...prev, blankContestant(prev.length)]);
  }

  function removeContestant(id) {
    if (contestants.length <= 2) {
      setToast({ type: 'error', message: 'Arena requires at least 2 contestants.' });
      return;
    }
    setContestants(prev => prev.filter(c => c._id !== id));
  }

  const needsNvidiaKey = contestants.some(c => c.model_provider === 'nvidia_nim');
  const needsOpenrouterKey = contestants.some(c => c.model_provider === 'openrouter');

  const isNvidiaNimReady = !needsNvidiaKey
    || (nvidiaNimBaseUrl !== 'https://integrate.api.nvidia.com/v1')
    || (apiKeysSet.nvidia_nim_api_key_set || !!nvidiaNimKey);
  const isOpenrouterReady = !needsOpenrouterKey
    || apiKeysSet.openrouter_api_key_set || !!openrouterKey;
  const allModelsSet = contestants.every(c => !!getEffectiveModel(c));
  const canSubmit = isNvidiaNimReady && isOpenrouterReady && allModelsSet;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedDatasetId) { setToast({ type: 'error', message: 'Please select a dataset.' }); return; }

    const contestantPayloads = contestants.map(c => ({
      label: c.label || getEffectiveModel(c),
      model_provider: c.model_provider,
      model_name: getEffectiveModel(c),
      temperature: parseFloat(c.temperature),
      system_prompt: c.system_prompt || '',
      max_tokens: c.max_tokens !== '' ? parseInt(c.max_tokens) : null,
      top_p: c.top_p !== '' ? parseFloat(c.top_p) : null,
      frequency_penalty: c.frequency_penalty !== '' ? parseFloat(c.frequency_penalty) : null,
      presence_penalty: c.presence_penalty !== '' ? parseFloat(c.presence_penalty) : null,
      multi_turn_history_mode: c.multi_turn_history_mode,
    }));

    setIsRunning(true);
    try {
      const body = {
        run_name: runName || 'Arena Run',
        dataset_id: selectedDatasetId,
        contestants: contestantPayloads,
        nvidia_nim_base_url: needsNvidiaKey ? (nvidiaNimBaseUrl || null) : null,
        nvidia_nim_api_key: needsNvidiaKey ? (nvidiaNimKey || null) : null,
        openrouter_api_key: needsOpenrouterKey ? (openrouterKey || null) : null,
      };

      const res = await fetch('/api/arena-runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Arena run failed.');
      }

      const data = await res.json();
      const caseCount = datasets.find(d => d.id === selectedDatasetId)?.cases?.length || 0;
      setToast({ type: 'success', message: `Arena completed! ${contestants.length} models judged on ${caseCount} cases.` });
      onArenaComplete(data.id);
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    } finally {
      setIsRunning(false);
    }
  };

  const CONTESTANT_COLORS = ['#00f2fe', '#bf55ec', '#10b981', '#f59e0b', '#ef4444'];

  return (
    <div className="fade-in">
      <div className="header-container">
        <div>
          <h1 className="page-title">Arena Evaluation</h1>
          <p className="page-subtitle">Run multiple models head-to-head on the same dataset. An LLM judge crowns the winner per case.</p>
        </div>
      </div>

      {datasets.length === 0 ? (
        <div className="glass-card" style={{ padding: '40px', textAlign: 'center' }}>
          <AlertTriangle size={32} className="text-warning" style={{ marginBottom: '12px' }} />
          <p>No datasets found. Create or upload one in Dataset Manager first.</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit}>
          <div className="dashboard-grid" style={{ gridTemplateColumns: '2fr 1fr' }}>

            {/* ── Left: Arena config ── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

              {/* Run name + dataset */}
              <div className="glass-card">
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Settings size={18} className="text-secondary" /> Arena Settings
                </h3>
                <div className="form-row">
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Arena Run Name</label>
                    <input type="text" className="form-control" value={runName} onChange={e => setRunName(e.target.value)} required />
                  </div>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Golden Q&amp;A Dataset</label>
                    <select className="form-select" value={selectedDatasetId} onChange={e => setSelectedDatasetId(e.target.value)}>
                      {datasets.map(d => (
                        <option key={d.id} value={d.id}>{d.name} ({d.cases?.length || 0} cases)</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Contestant cards */}
              {contestants.map((c, idx) => (
                <div key={c._id} className="glass-card" style={{ borderColor: CONTESTANT_COLORS[idx % CONTESTANT_COLORS.length] + '44' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                    <h3 style={{ fontSize: '15px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{
                        width: '22px', height: '22px', borderRadius: '50%', display: 'inline-flex',
                        alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: 700,
                        background: CONTESTANT_COLORS[idx % CONTESTANT_COLORS.length], color: '#000'
                      }}>{idx + 1}</span>
                      Contestant {idx + 1}
                    </h3>
                    {contestants.length > 2 && (
                      <button type="button" onClick={() => removeContestant(c._id)}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-danger)', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px' }}>
                        <Trash2 size={14} /> Remove
                      </button>
                    )}
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Display Label (optional)</label>
                      <input type="text" className="form-control" placeholder={`e.g. Llama 3.1 8B`}
                        value={c.label} onChange={e => updateContestant(c._id, 'label', e.target.value)} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Provider</label>
                      <select className="form-select" value={c.model_provider} onChange={e => updateContestant(c._id, 'model_provider', e.target.value)}>
                        <option value="mock">Sandbox (Mock)</option>
                        <option value="nvidia_nim">Nvidia NIM</option>
                        <option value="openrouter">OpenRouter</option>
                      </select>
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Model</label>
                      {c.model_provider === 'mock' && (
                        <select className="form-select" value={c.model_name} onChange={e => updateContestant(c._id, 'model_name', e.target.value)}>
                          <option value="llama-3.2-3b-mock">Llama-3.2-3B (Mock)</option>
                          <option value="llama-3.2-11b-mock">Llama-3.2-11B (Mock)</option>
                          <option value="phi-4-mini-mock">Phi-4-mini (Mock)</option>
                          <option value="mistral-7b-mock">Mistral-7B (Mock)</option>
                        </select>
                      )}
                      {c.model_provider === 'nvidia_nim' && (
                        <input type="text" className="form-control" placeholder="e.g. meta/llama-3.1-8b-instruct"
                          value={c.nvidiaNimModel} onChange={e => updateContestant(c._id, 'nvidiaNimModel', e.target.value)} required />
                      )}
                      {c.model_provider === 'openrouter' && (
                        <input type="text" className="form-control" placeholder="e.g. meta-llama/llama-3.1-8b-instruct:free"
                          value={c.openrouterModel} onChange={e => updateContestant(c._id, 'openrouterModel', e.target.value)} required />
                      )}
                    </div>
                    <div className="form-group">
                      <label className="form-label">Temperature: {c.temperature}</label>
                      <input type="range" min="0" max="1.5" step="0.1" className="form-control"
                        style={{ height: '38px', padding: '0 8px' }}
                        value={c.temperature} onChange={e => updateContestant(c._id, 'temperature', e.target.value)} />
                    </div>
                  </div>

                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">System Prompt (optional)</label>
                    <textarea className="form-control form-textarea" style={{ minHeight: '60px' }}
                      placeholder="Custom instructions for this contestant…"
                      value={c.system_prompt} onChange={e => updateContestant(c._id, 'system_prompt', e.target.value)} />
                  </div>
                </div>
              ))}

              {/* Add contestant button */}
              {contestants.length < 5 && (
                <button type="button" onClick={addContestant}
                  className="btn btn-secondary"
                  style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%', justifyContent: 'center', padding: '14px', borderStyle: 'dashed' }}>
                  <Plus size={16} /> Add Contestant ({contestants.length}/5)
                </button>
              )}

              {/* Submit row */}
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                {isRunning ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-muted)', fontSize: '13px' }}>
                    <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                    {loadingSteps[loadingStep]}
                  </div>
                ) : (
                  <button type="submit" className="btn btn-primary" style={{ minWidth: '180px' }} disabled={!canSubmit}>
                    <Swords size={16} /> Start Arena Run
                  </button>
                )}
              </div>
            </div>

            {/* ── Right: credentials + info ── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div className="glass-card" style={{ height: 'fit-content' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Key size={18} className="text-secondary" /> Credentials
                </h3>

                {[
                  { label: 'Nvidia NIM', needed: needsNvidiaKey, envSet: apiKeysSet.nvidia_nim_api_key_set, keyVal: nvidiaNimKey },
                  { label: 'OpenRouter', needed: needsOpenrouterKey, envSet: apiKeysSet.openrouter_api_key_set, keyVal: openrouterKey },
                ].map(row => (
                  <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                    <div>
                      <p style={{ fontWeight: 500, fontSize: '13px' }}>{row.label}</p>
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        {!row.needed ? 'Not used' : row.envSet ? 'Loaded from env' : 'Needs key input below'}
                      </p>
                    </div>
                    {!row.needed
                      ? <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>—</span>
                      : (row.envSet || !!row.keyVal)
                        ? <ShieldCheck size={20} style={{ color: 'var(--color-success)' }} />
                        : <span className="status-dot inactive" style={{ width: '12px', height: '12px' }}></span>
                    }
                  </div>
                ))}

                {needsNvidiaKey && (
                  <>
                    <div className="form-group fade-in" style={{ padding: '12px', background: 'rgba(99,102,241,0.05)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 'var(--radius-md)', marginBottom: '10px' }}>
                      <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
                        <Server size={12} /> NIM Base URL
                      </label>
                      <input type="text" className="form-control" value={nvidiaNimBaseUrl} onChange={e => setNvidiaNimBaseUrl(e.target.value)} />
                    </div>
                    {!apiKeysSet.nvidia_nim_api_key_set && nvidiaNimBaseUrl === 'https://integrate.api.nvidia.com/v1' && (
                      <div className="form-group fade-in" style={{ padding: '12px', background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 'var(--radius-md)' }}>
                        <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
                          <Key size={12} /> Nvidia NIM API Key
                        </label>
                        <input type="password" className="form-control" placeholder="NVIDIA_NIM_API_KEY"
                          value={nvidiaNimKey} onChange={e => setNvidiaNimKey(e.target.value)} />
                      </div>
                    )}
                  </>
                )}

                {needsOpenrouterKey && !apiKeysSet.openrouter_api_key_set && (
                  <div className="form-group fade-in" style={{ padding: '12px', background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 'var(--radius-md)' }}>
                    <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
                      <Key size={12} /> OpenRouter API Key
                    </label>
                    <input type="password" className="form-control" placeholder="OPENROUTER_API_KEY"
                      value={openrouterKey} onChange={e => setOpenrouterKey(e.target.value)} />
                    <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>
                      <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer" style={{ color: 'var(--color-primary)' }}>Get a free key →</a>
                    </p>
                  </div>
                )}
              </div>

              <div className="glass-card" style={{ height: 'fit-content', fontSize: '13px', color: 'var(--text-muted)', lineHeight: '1.7' }}>
                <p style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Swords size={14} /> How Arena Works
                </p>
                <ol style={{ paddingLeft: '16px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <li>Each contestant model answers every case in the dataset.</li>
                  <li>An LLM judge scores all answers on correctness, completeness, and clarity.</li>
                  <li>A pairwise judge picks the best response per case (or declares a tie).</li>
                  <li>Win rates and aggregate scores form the final leaderboard.</li>
                </ol>
              </div>
            </div>
          </div>
        </form>
      )}
    </div>
  );
}
