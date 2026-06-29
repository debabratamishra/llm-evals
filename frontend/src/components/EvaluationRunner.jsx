import React, { useState, useEffect } from 'react';
import { Play, Settings, AlertTriangle, Key, ShieldCheck, Loader2, Server, Globe, ExternalLink } from 'lucide-react';


export default function EvaluationRunner({ datasets, apiKeysSet, onRunComplete, setToast }) {
  const [runName,           setRunName]           = useState('');
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [provider,          setProvider]          = useState('mock');
  const [modelName,         setModelName]         = useState('llama-3.2-3b-mock');
  const [temperature,       setTemperature]       = useState(0.2);
  const [systemPrompt,      setSystemPrompt]      = useState('');

  // ── per-provider credential/config inputs ──
  const [ollamaBaseUrlInput,    setOllamaBaseUrlInput]    = useState('http://localhost:11434');
  const [ollamaCloudUrlInput,   setOllamaCloudUrlInput]   = useState('');
  const [ollamaCloudKeyInput,   setOllamaCloudKeyInput]   = useState('');
  const [openrouterKeyInput,    setOpenrouterKeyInput]    = useState('');
  const [openrouterCustomModel, setOpenrouterCustomModel] = useState('');
  const [ollamaModelInput,      setOllamaModelInput]      = useState('llama3.2');

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

  // Pre-fill Ollama base URL from env-detected value
  useEffect(() => {
    if (apiKeysSet.ollama_base_url && apiKeysSet.ollama_base_url !== 'http://localhost:11434') {
      setOllamaBaseUrlInput(apiKeysSet.ollama_base_url);
      setOllamaCloudUrlInput(apiKeysSet.ollama_base_url);
    }
  }, [apiKeysSet.ollama_base_url]);

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
  }, [selectedDatasetId, provider, modelName, openrouterCustomModel, ollamaModelInput, datasets]);

  // Reset model defaults when provider changes
  useEffect(() => {
    if (provider === 'ollama' || provider === 'ollama_cloud') setModelName(ollamaModelInput);
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
    if (provider === 'ollama' || provider === 'ollama_cloud') return ollamaModelInput;
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
    if ((provider === 'ollama' || provider === 'ollama_cloud') && !ollamaModelInput.trim()) {
      setToast({ type: 'error', message: 'Please enter an Ollama model name.' });
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
        ollama_base_url:    provider === 'ollama'       ? (ollamaBaseUrlInput  || null)
                          : provider === 'ollama_cloud' ? (ollamaCloudUrlInput || null) : null,
        ollama_api_key:     provider === 'ollama_cloud' ? (ollamaCloudKeyInput || null) : null,
        openrouter_api_key: provider === 'openrouter'   ? (openrouterKeyInput  || null) : null,
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
  const isOllamaCloudReady = provider !== 'ollama_cloud'
    || (!!ollamaCloudUrlInput && (apiKeysSet.ollama_api_key_set || !!ollamaCloudKeyInput));
  const isOpenrouterReady  = provider !== 'openrouter'
    || apiKeysSet.openrouter_api_key_set || !!openrouterKeyInput;
  const canSubmit = isOllamaCloudReady && isOpenrouterReady;

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
      label:  'Ollama Local',
      detail: 'No key required · uses base URL',
      ready:  true,
    },
    {
      label:  'Ollama Cloud',
      detail: apiKeysSet.ollama_api_key_set ? 'Loaded via OLLAMA_API_KEY env' : 'Requires key + URL',
      ready:  apiKeysSet.ollama_api_key_set || !!ollamaCloudKeyInput,
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
                    <option value="ollama">Ollama - Local</option>
                    <option value="ollama_cloud">Ollama - Cloud</option>
                    <option value="openrouter">OpenRouter</option>
                  </select>
                </div>
              </div>

              {/* Model selector — varies by provider */}
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Target Model</label>

                  {(provider === 'ollama' || provider === 'ollama_cloud') && (
                    <input
                      type="text"
                      className="form-control"
                      placeholder="e.g. llama3.2, mistral, phi4"
                      value={ollamaModelInput}
                      onChange={e => setOllamaModelInput(e.target.value)}
                      required
                    />
                  )}

                  {provider === 'openrouter' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <input
                        type="text"
                        className="form-control"
                        placeholder="e.g. meta-llama/llama-3.1-8b-instruct:free"
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

              {/* ── Provider-specific credential / config inputs ── */}

              {provider === 'ollama' && (
                <UrlInputBox
                  label="Ollama Base URL"
                  placeholder="http://localhost:11434"
                  value={ollamaBaseUrlInput}
                  onChange={setOllamaBaseUrlInput}
                  note="Point this at your local Ollama instance. Default: http://localhost:11434"
                />
              )}

              {provider === 'ollama_cloud' && (
                <>
                  <UrlInputBox
                    label="Ollama Cloud Base URL"
                    placeholder="https://your-ollama-cloud-host.example.com"
                    value={ollamaCloudUrlInput}
                    onChange={setOllamaCloudUrlInput}
                    note="The base URL of your cloud-hosted Ollama-compatible endpoint."
                  />
                  {!apiKeysSet.ollama_api_key_set && (
                    <KeyInputBox
                      label="Ollama Cloud API Key"
                      placeholder="Enter your cloud Ollama API key (not saved on server)"
                      value={ollamaCloudKeyInput}
                      onChange={setOllamaCloudKeyInput}
                      note="No OLLAMA_API_KEY env variable detected. Key is used only for this request."
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
          {provider === 'ollama' && (
            <div className="fade-in" style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
              <p style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, marginBottom: '6px' }}>
                <Server size={13} /> Ollama Local
              </p>
              Runs fully offline. Make sure <code>ollama serve</code> is running and the model is pulled locally (<code>ollama pull {ollamaModelInput}</code>).
            </div>
          )}
          {provider === 'ollama_cloud' && (
            <div className="fade-in" style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
              <p style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, marginBottom: '6px' }}>
                <Globe size={13} /> Ollama Cloud
              </p>
              Points at any OpenAI-compatible cloud endpoint running Ollama — e.g. a self-hosted VM, RunPod, or similar. Requires a base URL and usually an API key.
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
