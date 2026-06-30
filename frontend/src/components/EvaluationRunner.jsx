import React, { useState, useEffect } from 'react';
import { Play, Settings, AlertTriangle, Key, ShieldCheck, Loader2, Server, ExternalLink } from 'lucide-react';


export default function EvaluationRunner({ datasets, apiKeysSet, onRunComplete, setToast }) {
  const [runName,           setRunName]           = useState('');
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [provider,          setProvider]          = useState('mock');
  const [modelName,         setModelName]         = useState('llama-3.2-3b-mock');
  const [temperature,       setTemperature]       = useState(0.2);
  const [systemPrompt,      setSystemPrompt]      = useState('');
  
  // Advanced parameters
  const [showAdvanced,           setShowAdvanced]           = useState(false);
  const [maxTokens,              setMaxTokens]              = useState('');
  const [topP,                   setTopP]                   = useState('');
  const [frequencyPenalty,       setFrequencyPenalty]       = useState('');
  const [presencePenalty,        setPresencePenalty]        = useState('');
  const [historyMode,            setHistoryMode]            = useState('model_response');

  // ── per-provider credential/config inputs ──
  const [nvidiaNimBaseUrlInput, setNvidiaNimBaseUrlInput] = useState('https://integrate.api.nvidia.com/v1');
  const [nvidiaNimKeyInput,     setNvidiaNimKeyInput]     = useState('');
  const [openrouterKeyInput,    setOpenrouterKeyInput]    = useState('');
  const [openrouterCustomModel, setOpenrouterCustomModel] = useState('');
  const [nvidiaNimModelInput,   setNvidiaNimModelInput]   = useState('meta/llama-3.1-8b-instruct');

  const [isRunning,   setIsRunning]   = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);

  const loadingSteps = [
    'Spinning up target environment…',
    'Sending input questions to model…',
    'Receiving and recording candidate responses…',
    'Running LLM-as-a-Judge evaluations…',
    'Calculating semantic similarity & exact matches…',
    'Analysing costs and latencies…',
    'Aggregating benchmark results…',
  ];

  // Pre-fill Nvidia NIM base URL from env-detected value
  useEffect(() => {
    if (apiKeysSet.nvidia_nim_base_url && apiKeysSet.nvidia_nim_base_url !== 'https://integrate.api.nvidia.com/v1') {
      setNvidiaNimBaseUrlInput(apiKeysSet.nvidia_nim_base_url);
    }
  }, [apiKeysSet.nvidia_nim_base_url]);

  useEffect(() => {
    if (datasets.length > 0 && !selectedDatasetId) setSelectedDatasetId(datasets[0].id);
  }, [datasets, selectedDatasetId]);

  // Auto-generate run name
  useEffect(() => {
    if (!selectedDatasetId) return;
    const ds = datasets.find(d => d.id === selectedDatasetId);
    const dsName = ds ? ds.name : 'Dataset';
    setRunName(`${getEffectiveModelName()} on ${dsName}`);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetId, provider, modelName, openrouterCustomModel, nvidiaNimModelInput, datasets]);

  // Reset model defaults when provider changes
  useEffect(() => {
    if (provider === 'nvidia_nim') setModelName(nvidiaNimModelInput);
    else if (provider === 'openrouter') setModelName(getOpenRouterModel());
    else setModelName('llama-3.2-3b-mock');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider]);

  // Loading step rotation
  useEffect(() => {
    let interval;
    if (isRunning) {
      setLoadingStep(0);
      interval = setInterval(() => setLoadingStep(p => (p + 1) % loadingSteps.length), 2500);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  function getOpenRouterModel() {
    return openrouterCustomModel;
  }

  function getEffectiveModelName() {
    if (provider === 'openrouter') return getOpenRouterModel();
    if (provider === 'nvidia_nim') return nvidiaNimModelInput;
    return modelName;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedDatasetId) {
      setToast({ type: 'error', message: 'Please select a dataset.' });
      return;
    }
    const effectiveModel = getEffectiveModelName();
    if (provider === 'openrouter' && !effectiveModel) {
      setToast({ type: 'error', message: 'Please select or enter an OpenRouter model ID.' });
      return;
    }
    if (provider === 'nvidia_nim' && !nvidiaNimModelInput.trim()) {
      setToast({ type: 'error', message: 'Please enter an Nvidia NIM model name.' });
      return;
    }

    const caseCount = datasets.find(d => d.id === selectedDatasetId)?.cases?.length || 0;
    setIsRunning(true);
    try {
      const body = {
        run_name:           runName || 'Evaluation Run',
        dataset_id:         selectedDatasetId,
        model_provider:     provider,
        model_name:         effectiveModel,
        temperature:        parseFloat(temperature),
        system_prompt:      systemPrompt,
        max_tokens:              maxTokens !== '' ? parseInt(maxTokens) : null,
        top_p:                   topP !== '' ? parseFloat(topP) : null,
        frequency_penalty:       frequencyPenalty !== '' ? parseFloat(frequencyPenalty) : null,
        presence_penalty:        presencePenalty !== '' ? parseFloat(presencePenalty) : null,
        multi_turn_history_mode: historyMode,
        nvidia_nim_base_url: provider === 'nvidia_nim' ? (nvidiaNimBaseUrlInput || null) : null,
        nvidia_nim_api_key:  provider === 'nvidia_nim' ? (nvidiaNimKeyInput     || null) : null,
        openrouter_api_key:  provider === 'openrouter' ? (openrouterKeyInput    || null) : null,
      };

      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Evaluation run failed.');
      }

      const data = await res.json();
      setToast({ type: 'success', message: `Evaluation completed! Scored ${caseCount} cases.` });
      onRunComplete(data.id);
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    } finally {
      setIsRunning(false);
    }
  };

  // Gate the submit button
  const isNvidiaNimReady = provider !== 'nvidia_nim'
    || (nvidiaNimBaseUrlInput !== 'https://integrate.api.nvidia.com/v1')
    || (apiKeysSet.nvidia_nim_api_key_set || !!nvidiaNimKeyInput);
  const isOpenrouterReady  = provider !== 'openrouter'
    || apiKeysSet.openrouter_api_key_set || !!openrouterKeyInput;
  const canSubmit = isNvidiaNimReady && isOpenrouterReady;

  // ── Reusable input boxes ─────────────────────────────────────────────

  const KeyInputBox = ({ label, placeholder, value, onChange, note }) => (
    <div className="form-group fade-in" style={{ padding: '16px', background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 'var(--radius-md)' }}>
      <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <Key size={14} className="text-warning" /> {label}
      </label>
      <input type="password" className="form-control" placeholder={placeholder} value={value} onChange={e => onChange(e.target.value)} required />
      {note && <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>{note}</p>}
    </div>
  );

  const UrlInputBox = ({ label, placeholder, value, onChange, note }) => (
    <div className="form-group fade-in" style={{ padding: '16px', background: 'rgba(99,102,241,0.05)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 'var(--radius-md)' }}>
      <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <Server size={14} className="text-secondary" /> {label}
      </label>
      <input type="text" className="form-control" placeholder={placeholder} value={value} onChange={e => onChange(e.target.value)} />
      {note && <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>{note}</p>}
    </div>
  );

  // ── Credential sidebar rows ──────────────────────────────────────────

  const credentialRows = [
    {
      label:  'Nvidia NIM',
      detail: apiKeysSet.nvidia_nim_api_key_set ? 'Loaded via NVIDIA_NIM_API_KEY env' : 'Requires key for cloud',
      ready:  apiKeysSet.nvidia_nim_api_key_set || !!nvidiaNimKeyInput || (!!nvidiaNimBaseUrlInput && nvidiaNimBaseUrlInput !== 'https://integrate.api.nvidia.com/v1'),
    },
    {
      label:  'OpenRouter',
      detail: apiKeysSet.openrouter_api_key_set ? 'Loaded via OPENROUTER_API_KEY env' : 'Requires key input',
      ready:  apiKeysSet.openrouter_api_key_set || !!openrouterKeyInput,
    },
  ];

  // ── Render ───────────────────────────────────────────────────────────

  return (
    <div className="fade-in">
      <div className="header-container">
        <div>
          <h1 className="page-title">Run Evaluation</h1>
          <p className="page-subtitle">Configure hyper-parameters, target models, and trigger evaluation pipelines.</p>
        </div>
      </div>

      <div className="dashboard-grid" style={{ gridTemplateColumns: '2fr 1fr' }}>

        {/* ── Config panel ── */}
        <div className="glass-card">
          <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Settings size={20} className="text-secondary" /> Eval Settings
          </h3>

          {datasets.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px' }}>
              <AlertTriangle size={32} className="text-warning" style={{ marginBottom: '12px' }} />
              <p>No datasets available. Please create or upload a dataset first in the Dataset Manager.</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit}>

              {/* Run name */}
              <div className="form-group">
                <label className="form-label">Run Reference Name</label>
                <input type="text" className="form-control"
                  placeholder="e.g. Llama-3.2 on Reasoning Benchmark"
                  value={runName} onChange={e => setRunName(e.target.value)} required />
              </div>

              {/* Dataset + Provider row */}
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Golden Q&amp;A Dataset</label>
                  <select className="form-select" value={selectedDatasetId} onChange={e => setSelectedDatasetId(e.target.value)}>
                    {datasets.map(d => (
                      <option key={d.id} value={d.id}>{d.name} ({d.cases?.length || 0} cases)</option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Model Provider</label>
                  <select className="form-select" value={provider} onChange={e => setProvider(e.target.value)}>
                    <option value="mock">Sandbox Evaluator (No Keys Required)</option>
                    <option value="nvidia_nim">Nvidia NIM</option>
                    <option value="openrouter">OpenRouter</option>
                  </select>
                </div>
              </div>

              {/* Model selector — varies by provider */}
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Target Model</label>

                  {provider === 'nvidia_nim' && (
                    <input
                      type="text"
                      className="form-control"
                      placeholder="e.g. meta/llama-3.2-3b-instruct"
                      value={nvidiaNimModelInput}
                      onChange={e => setNvidiaNimModelInput(e.target.value)}
                      required
                    />
                  )}

                  {provider === 'openrouter' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <input
                        type="text"
                        className="form-control"
                        placeholder="e.g. meta-llama/llama-3.2-3b-instruct"
                        value={openrouterCustomModel}
                        onChange={e => setOpenrouterCustomModel(e.target.value)}
                        required
                      />
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: 0 }}>
                        Browse all models at{' '}
                        <a href="https://openrouter.ai/models" target="_blank" rel="noreferrer"
                          style={{ color: 'var(--color-primary)', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                          openrouter.ai/models <ExternalLink size={10} />
                        </a>
                      </p>
                    </div>
                  )}

                  {provider === 'mock' && (
                    <select className="form-select" value={modelName} onChange={e => setModelName(e.target.value)}>
                      <option value="llama-3.2-3b-mock">meta-llama/Llama-3.2-3B-Instruct (Mock)</option>
                      <option value="llama-3.2-11b-mock">meta-llama/Llama-3.2-11B-Vision-Instruct (Mock)</option>
                      <option value="phi-4-mini-mock">microsoft/Phi-4-mini-instruct (Mock)</option>
                    </select>
                  )}
                </div>

                <div className="form-group">
                  <label className="form-label">Temperature: {temperature}</label>
                  <input type="range" min="0.0" max="1.5" step="0.1" className="form-control"
                    style={{ height: '38px', padding: '0 8px' }}
                    value={temperature} onChange={e => setTemperature(e.target.value)} />
                </div>
              </div>

              {/* System prompt */}
              <div className="form-group">
                <label className="form-label">System Instructions (Optional)</label>
                <textarea className="form-control form-textarea"
                  placeholder="You are a helpful assistant. Answer clearly and concisely."
                  value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} />
              </div>

              {/* Advanced Parameters Toggle */}
              <div style={{ marginBottom: '20px' }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', padding: '8px 12px' }}
                  onClick={() => setShowAdvanced(!showAdvanced)}
                >
                  <Settings size={14} /> 
                  {showAdvanced ? 'Hide Advanced Parameters' : 'Show Advanced Parameters'}
                </button>
              </div>

              {showAdvanced && (
                <div className="fade-in" style={{
                  padding: '20px',
                  background: 'rgba(255,255,255,0.01)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-md)',
                  marginBottom: '20px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '16px'
                }}>
                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Max Tokens (Optional)</label>
                      <input 
                        type="number" 
                        className="form-control" 
                        placeholder="e.g. 1024 (leave empty for default)"
                        value={maxTokens} 
                        onChange={e => setMaxTokens(e.target.value)} 
                        min="1"
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Top P: {topP !== '' ? topP : 'Default'}</label>
                      <input 
                        type="range" 
                        min="0.0" 
                        max="1.0" 
                        step="0.05" 
                        className="form-control"
                        style={{ height: '38px', padding: '0 8px' }}
                        value={topP === '' ? '1.0' : topP} 
                        onChange={e => setTopP(e.target.value)}
                      />
                      <button 
                        type="button" 
                        className="btn-link" 
                        style={{ fontSize: '11px', color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, marginTop: '4px', textAlign: 'left' }}
                        onClick={() => setTopP('')}
                      >
                        Reset to default
                      </button>
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Frequency Penalty: {frequencyPenalty !== '' ? frequencyPenalty : 'Default'}</label>
                      <input 
                        type="range" 
                        min="-2.0" 
                        max="2.0" 
                        step="0.1" 
                        className="form-control"
                        style={{ height: '38px', padding: '0 8px' }}
                        value={frequencyPenalty === '' ? '0.0' : frequencyPenalty} 
                        onChange={e => setFrequencyPenalty(e.target.value)}
                      />
                      <button 
                        type="button" 
                        className="btn-link" 
                        style={{ fontSize: '11px', color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, marginTop: '4px', textAlign: 'left' }}
                        onClick={() => setFrequencyPenalty('')}
                      >
                        Reset to default
                      </button>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Presence Penalty: {presencePenalty !== '' ? presencePenalty : 'Default'}</label>
                      <input 
                        type="range" 
                        min="-2.0" 
                        max="2.0" 
                        step="0.1" 
                        className="form-control"
                        style={{ height: '38px', padding: '0 8px' }}
                        value={presencePenalty === '' ? '0.0' : presencePenalty} 
                        onChange={e => setPresencePenalty(e.target.value)}
                      />
                      <button 
                        type="button" 
                        className="btn-link" 
                        style={{ fontSize: '11px', color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, marginTop: '4px', textAlign: 'left' }}
                        onClick={() => setPresencePenalty('')}
                      >
                        Reset to default
                      </button>
                    </div>
                  </div>

                  {/* Multi-turn history mode */}
                  <div className="form-group">
                    <label className="form-label" style={{ marginBottom: '8px' }}>
                      Multi-Turn History Mode
                    </label>
                    <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '10px', lineHeight: '1.5' }}>
                      Controls what is injected as the assistant turn in multi-turn conversation history.
                    </p>
                    <div style={{ display: 'flex', gap: '0', borderRadius: 'var(--radius-sm)', overflow: 'hidden', border: '1px solid var(--border-color)', width: 'fit-content' }}>
                      {[
                        { value: 'model_response', label: 'Model Response', desc: 'Use actual model output (realistic)' },
                        { value: 'ideal_response', label: 'Ideal Response', desc: 'Use golden answer (teacher forcing)' },
                      ].map(opt => (
                        <button
                          key={opt.value}
                          type="button"
                          title={opt.desc}
                          onClick={() => setHistoryMode(opt.value)}
                          style={{
                            padding: '8px 16px',
                            fontSize: '12px',
                            fontWeight: historyMode === opt.value ? 600 : 400,
                            background: historyMode === opt.value ? 'var(--color-primary)' : 'transparent',
                            color: historyMode === opt.value ? '#fff' : 'var(--text-secondary)',
                            border: 'none',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                          }}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                    <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>
                      {historyMode === 'model_response'
                        ? '⚡ Realistic: model errors in earlier turns cascade into later turns, measuring true chat robustness.'
                        : '📐 Teacher Forcing: each turn is evaluated against ideal context, isolating per-turn capability.'}
                    </p>
                  </div>
                </div>
              )}

              {/* ── Provider-specific credential / config inputs ── */}

              {provider === 'nvidia_nim' && (
                <>
                  <UrlInputBox
                    label="Nvidia NIM Base URL"
                    placeholder="https://integrate.api.nvidia.com/v1"
                    value={nvidiaNimBaseUrlInput}
                    onChange={setNvidiaNimBaseUrlInput}
                    note="Default: https://integrate.api.nvidia.com/v1. Override if self-hosting NIM."
                  />
                  {!apiKeysSet.nvidia_nim_api_key_set && (nvidiaNimBaseUrlInput === 'https://integrate.api.nvidia.com/v1') && (
                    <KeyInputBox
                      label="Nvidia NIM API Key"
                      placeholder="Enter your NVIDIA_NIM_API_KEY (not saved on server)"
                      value={nvidiaNimKeyInput}
                      onChange={setNvidiaNimKeyInput}
                      note="No NVIDIA_NIM_API_KEY env variable detected. Key is used only for this request."
                    />
                  )}
                </>
              )}

              {provider === 'openrouter' && !apiKeysSet.openrouter_api_key_set && (
                <KeyInputBox
                  label="OpenRouter API Key"
                  placeholder="Enter your OPENROUTER_API_KEY (not saved on server)"
                  value={openrouterKeyInput}
                  onChange={setOpenrouterKeyInput}
                  note="No environment variable detected. Get a free key at openrouter.ai. Key is used only for this request."
                />
              )}

              {/* Submit */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '24px' }}>
                {isRunning ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-muted)', fontSize: '13px' }}>
                    <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                    {loadingSteps[loadingStep]}
                  </div>
                ) : (
                  <button type="submit" className="btn btn-primary" style={{ minWidth: '160px' }} disabled={!canSubmit}>
                    <Play size={16} /> Execute Run
                  </button>
                )}
              </div>

            </form>
          )}
        </div>

        {/* ── Credentials sidebar ── */}
        <div className="glass-card" style={{ height: 'fit-content' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Key size={18} className="text-secondary" /> Credentials Status
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {credentialRows.map(row => (
              <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <p style={{ fontWeight: 500, fontSize: '13px' }}>{row.label}</p>
                  <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{row.detail}</p>
                </div>
                {row.ready
                  ? <ShieldCheck size={20} style={{ color: 'var(--color-success)' }} />
                  : <span className="status-dot inactive" style={{ width: '12px', height: '12px' }}></span>
                }
              </div>
            ))}
          </div>

          <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: '20px 0' }} />

          {/* Provider info blurb */}
          {provider === 'nvidia_nim' && (
            <div className="fade-in" style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
              <p style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, marginBottom: '6px' }}>
                <Server size={13} /> Nvidia NIM
              </p>
              Access models hosted on NVIDIA's cloud NIM, or configure a self-hosted NIM container. Default base URL: <code>https://integrate.api.nvidia.com/v1</code>.
            </div>
          )}
          {provider === 'openrouter' && (
            <div className="fade-in" style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
              <p style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, marginBottom: '6px' }}>
                <ExternalLink size={13} /> OpenRouter
              </p>
              Routes to 200+ models through a single API key. Many models have a free tier.{' '}
              <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer" style={{ color: 'var(--color-primary)' }}>
                Get a free key →
              </a>
            </div>
          )}
          {provider === 'mock' && (
            <div className="fade-in" style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
              Sandbox mode uses deterministic mock responses — no API keys or network access required. Great for testing the evaluation pipeline.
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
