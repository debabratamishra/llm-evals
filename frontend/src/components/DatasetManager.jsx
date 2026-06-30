import React, { useState, useRef, useEffect } from 'react';
import { Database, Plus, Trash2, Eye, UploadCloud, FileText, CheckCircle, X, ShieldAlert, Layers, RefreshCw } from 'lucide-react';

export default function DatasetManager({ datasets, onRefresh, setToast }) {
  const [viewCasesDataset, setViewCasesDataset] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [uploadMode, setUploadMode] = useState('file'); // 'file', 'manual', 'hf'
  
  // File upload state
  const [uploadName, setUploadName] = useState('');
  const [uploadDesc, setUploadDesc] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  // Manual creation state
  const [manualName, setManualName] = useState('');
  const [manualDesc, setManualDesc] = useState('');
  const [datasetType, setDatasetType] = useState('single'); // 'single', 'multi'
  const [manualCases, setManualCases] = useState([{ question: '', ideal_answer: '' }]);

  // Hugging Face state
  const [hfPath, setHfPath] = useState('');
  const [hfLoading, setHfLoading] = useState(false);
  const [hfData, setHfData] = useState(null); // { configs, splits, columns, preview_rows }
  
  const [hfConfig, setHfConfig] = useState('');
  const [hfSplit, setHfSplit] = useState('');
  const [questionCol, setQuestionCol] = useState('');
  const [answerCol, setAnswerCol] = useState('');
  const [choicesCol, setChoicesCol] = useState('');
  const [hfLimit, setHfLimit] = useState(50);
  const [hfDatasetName, setHfDatasetName] = useState('');
  const [hfDatasetDesc, setHfDatasetDesc] = useState('');

  const resetUploadForm = () => {
    setUploadName('');
    setUploadDesc('');
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const resetManualForm = () => {
    setManualName('');
    setManualDesc('');
    setDatasetType('single');
    setManualCases([{ question: '', ideal_answer: '' }]);
  };

  const resetHfForm = () => {
    setHfPath('');
    setHfData(null);
    setHfConfig('');
    setHfSplit('');
    setQuestionCol('');
    setAnswerCol('');
    setChoicesCol('');
    setHfLimit(50);
    setHfDatasetName('');
    setHfDatasetDesc('');
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      if (!uploadName) {
        const baseName = file.name.substring(0, file.name.lastIndexOf('.')) || file.name;
        setUploadName(baseName.replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase()));
      }
    }
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setToast({ type: 'error', message: 'Please select a file to upload' });
      return;
    }

    const formData = new FormData();
    formData.append('name', uploadName);
    formData.append('description', uploadDesc);
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/api/datasets/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Failed to upload dataset');
      }

      setToast({ type: 'success', message: 'Dataset uploaded and processed successfully' });
      resetUploadForm();
      setIsCreating(false);
      onRefresh();
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    }
  };

  const handleDatasetTypeChange = (type) => {
    setDatasetType(type);
    if (type === 'single') {
      setManualCases([{ question: '', ideal_answer: '' }]);
    } else {
      setManualCases([{ turns: [{ user_message: '', ideal_response: '' }] }]);
    }
  };

  const handleManualAddRow = () => {
    if (datasetType === 'single') {
      setManualCases([...manualCases, { question: '', ideal_answer: '' }]);
    } else {
      setManualCases([...manualCases, { turns: [{ user_message: '', ideal_response: '' }] }]);
    }
  };

  const handleManualRemoveRow = (idx) => {
    if (manualCases.length === 1) return;
    setManualCases(manualCases.filter((_, i) => i !== idx));
  };

  const handleManualCaseChange = (idx, field, val) => {
    const updated = [...manualCases];
    updated[idx][field] = val;
    setManualCases(updated);
  };

  const handleAddTurn = (caseIdx) => {
    const updated = [...manualCases];
    updated[caseIdx].turns.push({ user_message: '', ideal_response: '' });
    setManualCases(updated);
  };

  const handleRemoveTurn = (caseIdx, turnIdx) => {
    const updated = [...manualCases];
    if (updated[caseIdx].turns.length === 1) return;
    updated[caseIdx].turns = updated[caseIdx].turns.filter((_, i) => i !== turnIdx);
    setManualCases(updated);
  };

  const handleTurnChange = (caseIdx, turnIdx, field, val) => {
    const updated = [...manualCases];
    updated[caseIdx].turns[turnIdx][field] = val;
    setManualCases(updated);
  };

  const handleManualSubmit = async (e) => {
    e.preventDefault();
    if (!manualName.trim()) {
      setToast({ type: 'error', message: 'Dataset name is required' });
      return;
    }

    let filteredCases = [];
    if (datasetType === 'single') {
      filteredCases = manualCases.filter(c => c.question.trim() && c.ideal_answer.trim());
      if (filteredCases.length === 0) {
        setToast({ type: 'error', message: 'Dataset must contain at least one valid Q&A case' });
        return;
      }
    } else {
      // Validate multi-turn
      filteredCases = manualCases.map(c => {
        const validTurns = c.turns.filter(t => t.user_message.trim() && t.ideal_response.trim());
        return { ...c, turns: validTurns };
      }).filter(c => c.turns.length > 0);

      if (filteredCases.length === 0) {
        setToast({ type: 'error', message: 'Dataset must contain at least one valid multi-turn case with at least one turn' });
        return;
      }
    }

    try {
      const res = await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: manualName,
          description: manualDesc,
          cases: filteredCases
        }),
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Failed to create dataset');
      }

      setToast({ type: 'success', message: 'Manual dataset created successfully' });
      resetManualForm();
      setIsCreating(false);
      onRefresh();
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    }
  };

  // Hugging Face Integration
  const handleInspectHf = async () => {
    if (!hfPath.trim()) {
      setToast({ type: 'error', message: 'Hugging Face Dataset Path is required (e.g. cais/mmlu)' });
      return;
    }

    setHfLoading(true);
    setHfData(null);
    try {
      const res = await fetch(`/api/datasets/preview-hf?path=${encodeURIComponent(hfPath.trim())}`);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to load dataset details');
      }
      const data = await res.json();
      setHfData(data);
      
      // Auto select first configuration and split
      if (data.configs?.length > 0) setHfConfig(data.configs[0]);
      if (data.splits?.length > 0) {
        const validationSplit = data.splits.find(s => s.toLowerCase().includes('val') || s.toLowerCase().includes('dev'));
        setHfSplit(validationSplit || data.splits[0]);
      }
      
      // Attempt auto-matching columns
      const cols = data.columns || [];
      const qCol = cols.find(c => ['question', 'prompt', 'query', 'instruction', 'input', 'text'].includes(c.toLowerCase()));
      const aCol = cols.find(c => ['answer', 'ideal_answer', 'reference', 'target', 'gold', 'label'].includes(c.toLowerCase()));
      const cCol = cols.find(c => ['choices', 'options'].includes(c.toLowerCase()));
      
      if (qCol) setQuestionCol(qCol);
      if (aCol) setAnswerCol(aCol);
      if (cCol) setChoicesCol(cCol);

      // Pre-fill dataset name
      const cleanName = hfPath.split('/').pop().replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      setHfDatasetName(`${cleanName} HF Import`);
      
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    } finally {
      setHfLoading(false);
    }
  };

  const handleImportHfSubmit = async (e) => {
    e.preventDefault();
    if (!hfPath.trim() || !questionCol || !answerCol || !hfDatasetName) {
      setToast({ type: 'error', message: 'Missing required configuration parameters.' });
      return;
    }

    setHfLoading(true);
    try {
      const res = await fetch('/api/datasets/import-hf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: hfPath.trim(),
          config: hfConfig || null,
          split: hfSplit,
          question_column: questionCol,
          answer_column: answerCol,
          choices_column: choicesCol || null,
          limit: parseInt(hfLimit),
          dataset_name: hfDatasetName,
          dataset_description: hfDatasetDesc
        })
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Import failed');
      }

      setToast({ type: 'success', message: `Imported HuggingFace dataset "${hfDatasetName}" successfully!` });
      resetHfForm();
      setIsCreating(false);
      onRefresh();
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    } finally {
      setHfLoading(false);
    }
  };

  const handleDelete = async (id, name) => {
    if (!confirm(`Are you sure you want to delete the dataset "${name}"?`)) return;

    try {
      const res = await fetch(`/api/datasets/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete dataset');
      
      setToast({ type: 'success', message: `Dataset "${name}" deleted` });
      onRefresh();
    } catch (err) {
      setToast({ type: 'error', message: err.message });
    }
  };

  return (
    <div className="fade-in">
      <div className="header-container">
        <div>
          <h1 className="page-title">Dataset Manager</h1>
          <p className="page-subtitle">Upload CSV/JSONs, define custom logic test suites, or stream datasets from the Hugging Face Hub.</p>
        </div>
        {!isCreating && (
          <button className="btn btn-primary" onClick={() => setIsCreating(true)}>
            <Plus size={16} /> Create Dataset
          </button>
        )}
      </div>

      {isCreating && (
        <div className="glass-card fade-in" style={{ marginBottom: '32px' }}>
          <div className="card-header" style={{ marginBottom: '24px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 600 }}>Create New Golden Dataset</h3>
            <button className="btn btn-secondary" style={{ padding: '6px 12px' }} onClick={() => setIsCreating(false)}>
              Cancel
            </button>
          </div>

          {/* Creation modes selector tabs */}
          <div style={{ display: 'flex', gap: '24px', marginBottom: '24px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
            <span 
              style={{ cursor: 'pointer', fontWeight: 600, fontSize: '14px', color: uploadMode === 'file' ? 'var(--color-primary)' : 'var(--text-secondary)', paddingBottom: '8px', borderBottom: uploadMode === 'file' ? '2px solid var(--color-primary)' : '' }}
              onClick={() => setUploadMode('file')}
            >
              Upload File (JSON/CSV)
            </span>
            <span 
              style={{ cursor: 'pointer', fontWeight: 600, fontSize: '14px', color: uploadMode === 'manual' ? 'var(--color-primary)' : 'var(--text-secondary)', paddingBottom: '8px', borderBottom: uploadMode === 'manual' ? '2px solid var(--color-primary)' : '' }}
              onClick={() => setUploadMode('manual')}
            >
              Define Manually
            </span>
            <span 
              style={{ cursor: 'pointer', fontWeight: 600, fontSize: '14px', color: uploadMode === 'hf' ? 'var(--color-primary)' : 'var(--text-secondary)', paddingBottom: '8px', borderBottom: uploadMode === 'hf' ? '2px solid var(--color-primary)' : '' }}
              onClick={() => setUploadMode('hf')}
            >
              Import from Hugging Face Hub
            </span>
          </div>

          {/* Mode 1: File Upload */}
          {uploadMode === 'file' && (
            <form onSubmit={handleUploadSubmit}>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Dataset Name</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    placeholder="e.g., Coding Syntax Check"
                    value={uploadName}
                    onChange={(e) => setUploadName(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Description (Optional)</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    placeholder="Brief description of evaluation target"
                    value={uploadDesc}
                    onChange={(e) => setUploadDesc(e.target.value)}
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Golden Q&A File (.json or .csv)</label>
                <div 
                  className="upload-zone"
                  onClick={() => fileInputRef.current && fileInputRef.current.click()}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    style={{ display: 'none' }} 
                    accept=".json,.csv"
                    onChange={handleFileChange}
                  />
                  <UploadCloud className="upload-icon" />
                  {selectedFile ? (
                    <div>
                      <p style={{ fontWeight: 600, color: 'var(--color-primary)' }}>{selectedFile.name}</p>
                      <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{(selectedFile.size / 1024).toFixed(2)} KB</p>
                    </div>
                  ) : (
                    <div>
                      <p style={{ fontWeight: 600 }}>Click to browse or drag your file here</p>
                      <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Supports JSON (list of question/ideal_answer objects) and CSV files</p>
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '12px' }}>
                <button type="button" className="btn btn-secondary" onClick={resetUploadForm}>Clear</button>
                <button type="submit" className="btn btn-primary">Process and Save</button>
              </div>
            </form>
          )}

          {/* Mode 2: Manual creation */}
          {uploadMode === 'manual' && (
            <form onSubmit={handleManualSubmit}>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Dataset Name</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    placeholder="e.g., General Logic Reasoning"
                    value={manualName}
                    onChange={(e) => setManualName(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Description (Optional)</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    placeholder="Context or criteria being evaluated"
                    value={manualDesc}
                    onChange={(e) => setManualDesc(e.target.value)}
                  />
                </div>
              </div>

              <div className="form-group" style={{ marginBottom: '24px' }}>
                <label className="form-label" style={{ fontWeight: 600 }}>Dataset Type</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button 
                    type="button" 
                    className={`btn ${datasetType === 'single' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => handleDatasetTypeChange('single')}
                    style={{ flex: 1, padding: '10px' }}
                  >
                    Single-Turn (Q&amp;A)
                  </button>
                  <button 
                    type="button" 
                    className={`btn ${datasetType === 'multi' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => handleDatasetTypeChange('multi')}
                    style={{ flex: 1, padding: '10px' }}
                  >
                    Multi-Turn Conversation
                  </button>
                </div>
              </div>

              <h4 style={{ marginBottom: '16px', fontSize: '14px', color: 'var(--text-secondary)' }}>Test Cases</h4>
              
              {datasetType === 'single' ? (
                manualCases.map((c, idx) => (
                  <div key={idx} style={{ display: 'flex', gap: '12px', marginBottom: '16px', alignItems: 'flex-start' }} className="fade-in">
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '14px', width: '20px' }}>{idx + 1}</span>
                    <div style={{ flex: 1 }}>
                      <textarea 
                        className="form-control form-textarea" 
                        placeholder="Prompt / Question"
                        value={c.question}
                        onChange={(e) => handleManualCaseChange(idx, 'question', e.target.value)}
                        required
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <textarea 
                        className="form-control form-textarea" 
                        placeholder="Ideal Golden Answer"
                        value={c.ideal_answer}
                        onChange={(e) => handleManualCaseChange(idx, 'ideal_answer', e.target.value)}
                        required
                      />
                    </div>
                    <button 
                      type="button" 
                      className="btn btn-secondary" 
                      style={{ padding: '12px', marginTop: '4px' }}
                      onClick={() => handleManualRemoveRow(idx)}
                      disabled={manualCases.length === 1}
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))
              ) : (
                manualCases.map((c, caseIdx) => (
                  <div key={caseIdx} style={{
                    marginBottom: '24px', 
                    padding: '20px', 
                    backgroundColor: 'rgba(255,255,255,0.02)', 
                    border: '1px solid var(--border-color)', 
                    borderRadius: 'var(--radius-md)'
                  }} className="fade-in">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                      <h5 style={{ fontWeight: 600, fontSize: '14px', margin: 0, color: 'var(--color-primary)' }}>
                        CASE #{caseIdx + 1}
                      </h5>
                      <button 
                        type="button" 
                        className="btn btn-secondary" 
                        style={{ padding: '6px 12px', fontSize: '12px', color: 'var(--color-danger)' }}
                        onClick={() => handleManualRemoveRow(caseIdx)}
                        disabled={manualCases.length === 1}
                      >
                        Remove Case
                      </button>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      {c.turns?.map((turn, turnIdx) => (
                        <div key={turnIdx} style={{
                          padding: '16px',
                          backgroundColor: 'var(--bg-input)',
                          borderRadius: 'var(--radius-sm)',
                        }} className="fade-in">
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                              Turn #{turnIdx + 1}
                            </span>
                            <button 
                              type="button" 
                              className="btn btn-secondary" 
                              style={{ padding: '4px 8px', fontSize: '11px', color: 'var(--color-danger)' }}
                              onClick={() => handleRemoveTurn(caseIdx, turnIdx)}
                              disabled={c.turns.length === 1}
                            >
                              Remove Turn
                            </button>
                          </div>
                          <div style={{ display: 'flex', gap: '12px' }}>
                            <div style={{ flex: 1 }}>
                              <label className="form-label" style={{ fontSize: '11px' }}>User Prompt</label>
                              <textarea 
                                className="form-control form-textarea" 
                                placeholder="User Message"
                                value={turn.user_message}
                                onChange={(e) => handleTurnChange(caseIdx, turnIdx, 'user_message', e.target.value)}
                                required
                                style={{ height: '70px' }}
                              />
                            </div>
                            <div style={{ flex: 1 }}>
                              <label className="form-label" style={{ fontSize: '11px' }}>Golden Response</label>
                              <textarea 
                                className="form-control form-textarea" 
                                placeholder="Ideal Assistant Response"
                                value={turn.ideal_response}
                                onChange={(e) => handleTurnChange(caseIdx, turnIdx, 'ideal_response', e.target.value)}
                                required
                                style={{ height: '70px' }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    <button 
                      type="button" 
                      className="btn btn-secondary" 
                      style={{ marginTop: '14px', padding: '8px 14px', fontSize: '12px' }}
                      onClick={() => handleAddTurn(caseIdx)}
                    >
                      <Plus size={14} /> Add Turn
                    </button>
                  </div>
                ))
              )}

              <button type="button" className="btn btn-secondary" style={{ marginTop: '12px' }} onClick={handleManualAddRow}>
                <Plus size={16} /> Add Test Case
              </button>

              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '24px' }}>
                <button type="button" className="btn btn-secondary" onClick={resetManualForm}>Reset Fields</button>
                <button type="submit" className="btn btn-primary">Save Dataset</button>
              </div>
            </form>
          )}

          {/* Mode 3: Hugging Face Import */}
          {uploadMode === 'hf' && (
            <div>
              <div className="form-group">
                <label className="form-label">Hugging Face Dataset Path</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <input 
                    type="text" 
                    className="form-control" 
                    placeholder="e.g., cais/mmlu, microsoft/ms_marco, truthful_qa, openbookqa"
                    value={hfPath}
                    onChange={(e) => setHfPath(e.target.value)}
                    disabled={hfLoading}
                  />
                  <button 
                    type="button" 
                    className="btn btn-secondary"
                    onClick={handleInspectHf}
                    disabled={hfLoading || !hfPath.trim()}
                    style={{ minWidth: '140px' }}
                  >
                    {hfLoading ? <RefreshCw className="spinner" size={16} /> : 'Inspect Dataset'}
                  </button>
                </div>
                <p style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginTop: '6px' }}>
                  Loads the dataset metadata streaming from Hugging Face Hub (huggingface.co/datasets).
                </p>
              </div>

              {hfLoading && !hfData && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '120px' }}>
                  <RefreshCw className="spinner text-primary" size={24} />
                  <span style={{ marginLeft: '10px', fontSize: '13px', color: 'var(--text-secondary)' }}>Downloading schema descriptors from Hugging Face...</span>
                </div>
              )}

              {hfData && (
                <form onSubmit={handleImportHfSubmit} className="fade-in">
                  <hr />
                  <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-primary)', marginBottom: '16px' }}>Schema Configuration</h4>
                  
                  <div className="form-row">
                    {hfData.configs?.length > 0 && (
                      <div className="form-group">
                        <label className="form-label">Sub-configuration (subset)</label>
                        <select className="form-select" value={hfConfig} onChange={e => setHfConfig(e.target.value)}>
                          {hfData.configs.map(cfg => <option key={cfg} value={cfg}>{cfg}</option>)}
                        </select>
                      </div>
                    )}
                    
                    <div className="form-group">
                      <label className="form-label">Dataset Split</label>
                      <select className="form-select" value={hfSplit} onChange={e => setHfSplit(e.target.value)}>
                        {hfData.splits.map(s => <option key={s} value={s}>{s}</option>)}
                      </select>
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Question Column</label>
                      <select className="form-select" value={questionCol} onChange={e => setQuestionCol(e.target.value)} required>
                        <option value="">-- Select Question Column --</option>
                        {hfData.columns.map(col => <option key={col} value={col}>{col}</option>)}
                      </select>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Ideal Answer Column</label>
                      <select className="form-select" value={answerCol} onChange={e => setAnswerCol(e.target.value)} required>
                        <option value="">-- Select Answer Column --</option>
                        {hfData.columns.map(col => <option key={col} value={col}>{col}</option>)}
                      </select>
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Choices Column (Optional - For Multiple Choice datasets)</label>
                      <select className="form-select" value={choicesCol} onChange={e => setChoicesCol(e.target.value)}>
                        <option value="">-- None (Standard Text QA) --</option>
                        {hfData.columns.map(col => <option key={col} value={col}>{col}</option>)}
                      </select>
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                        If selected, numeric answer indexes are automatically mapped to text labels.
                      </p>
                    </div>
                    
                    <div className="form-group">
                      <label className="form-label">Sample Limit (Max Rows to Import)</label>
                      <input 
                        type="number" 
                        className="form-control"
                        min="5"
                        max="200"
                        value={hfLimit}
                        onChange={e => setHfLimit(e.target.value)}
                        required
                      />
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                        Keep limits under 100 for fast eval execution.
                      </p>
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Saved Dataset Name</label>
                      <input 
                        type="text" 
                        className="form-control"
                        placeholder="e.g. MMLU Biology"
                        value={hfDatasetName}
                        onChange={e => setHfDatasetName(e.target.value)}
                        required
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Saved Dataset Description</label>
                      <input 
                        type="text" 
                        className="form-control"
                        placeholder="Brief summary..."
                        value={hfDatasetDesc}
                        onChange={e => setHfDatasetDesc(e.target.value)}
                      />
                    </div>
                  </div>

                  {/* Preview first item box */}
                  {hfData.preview_rows?.length > 0 && (
                    <div style={{ marginBottom: '20px' }}>
                      <label className="form-label">Row Preview (inspect headers & keys)</label>
                      <pre className="code-block" style={{ maxHeight: '150px', fontSize: '12px' }}>
                        {JSON.stringify(hfData.preview_rows[0], null, 2)}
                      </pre>
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '24px' }}>
                    <button type="button" className="btn btn-secondary" onClick={resetHfForm}>Reset</button>
                    <button type="submit" className="btn btn-primary" disabled={hfLoading}>
                      {hfLoading ? <RefreshCw className="spinner" size={16} /> : 'Download and Save'}
                    </button>
                  </div>
                </form>
              )}
            </div>
          )}
        </div>
      )}

      {/* Dataset Grid List */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '20px' }}>
        {datasets.map((dataset) => (
          <div key={dataset.id} className="glass-card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start', marginBottom: '16px' }}>
              <div className="card-icon-wrapper" style={{ backgroundColor: 'rgba(0, 242, 254, 0.05)', color: 'var(--color-primary)' }}>
                <Database size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: '16px', fontWeight: 600 }}>{dataset.name}</h3>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Created {new Date(dataset.created_at).toLocaleDateString()}</span>
              </div>
            </div>
            
            <p style={{ color: 'var(--text-secondary)', fontSize: '13px', flexGrow: 1, marginBottom: '20px' }}>
              {dataset.description || 'No description provided.'}
            </p>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-color)', paddingTop: '14px', marginTop: 'auto' }}>
              <span className="badge badge-info" style={{ fontSize: '11px' }}>
                {dataset.cases?.length || 0} Test Cases
              </span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button 
                  className="btn btn-secondary" 
                  style={{ padding: '8px 12px', fontSize: '13px' }}
                  onClick={() => setViewCasesDataset(dataset)}
                >
                  <Eye size={14} /> View
                </button>
                <button 
                  className="btn btn-secondary" 
                  style={{ padding: '8px 12px', fontSize: '13px', color: 'var(--color-danger)' }}
                  onClick={() => handleDelete(dataset.id, dataset.name)}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* View cases modal */}
      {viewCasesDataset && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 1000, padding: '20px'
        }} className="fade-in">
          <div className="glass-card" style={{ width: '100%', maxWidth: '800px', maxHeight: '80vh', display: 'flex', flexDirection: 'column' }}>
            <div className="card-header" style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '16px', marginBottom: '16px' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: 600 }}>{viewCasesDataset.name}</h3>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>List of golden references</p>
              </div>
              <button className="btn btn-secondary" style={{ padding: '6px 12px' }} onClick={() => setViewCasesDataset(null)}>
                <X size={16} />
              </button>
            </div>
            
            <div style={{ overflowY: 'auto', flexGrow: 1, paddingRight: '6px' }}>
              {viewCasesDataset.cases?.map((c, idx) => {
                const isMultiTurn = c.turns && c.turns.length > 0;
                return (
                  <div key={idx} style={{ marginBottom: '20px', padding: '16px', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontWeight: 600, color: 'var(--color-primary)', fontSize: '13px', marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
                      <span>CASE #{idx + 1} ({c.id})</span>
                      <span className="badge badge-info" style={{ fontSize: '10px' }}>
                        {isMultiTurn ? `${c.turns.length} Turns` : 'Single Turn'}
                      </span>
                    </div>
                    {isMultiTurn ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                        {c.turns.map((turn, tIdx) => (
                          <div key={tIdx} style={{ padding: '10px 14px', borderLeft: '3px solid var(--color-primary)', background: 'rgba(255,255,255,0.02)', borderRadius: '0 var(--radius-sm) var(--radius-sm) 0' }}>
                            <div style={{ fontWeight: 600, fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>TURN {tIdx + 1}</div>
                            <div style={{ marginBottom: '8px' }}>
                              <span style={{ color: 'var(--text-secondary)', fontSize: '11px', fontWeight: 600 }}>User: </span>
                              <span style={{ fontSize: '13px', whiteSpace: 'pre-wrap' }}>{turn.user_message}</span>
                            </div>
                            <div>
                              <span style={{ color: 'var(--text-secondary)', fontSize: '11px', fontWeight: 600 }}>Ideal Response: </span>
                              <span style={{ fontSize: '13px', whiteSpace: 'pre-wrap', color: 'var(--color-success)' }}>{turn.ideal_response}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <>
                        <div style={{ marginBottom: '10px' }}>
                          <div style={{ color: 'var(--text-secondary)', fontSize: '12px', fontWeight: 600, marginBottom: '2px' }}>Question / Prompt:</div>
                           <div style={{ fontSize: '14px', whiteSpace: 'pre-wrap' }}>{c.question}</div>
                         </div>
                         <div>
                           <div style={{ color: 'var(--text-secondary)', fontSize: '12px', fontWeight: 600, marginBottom: '2px' }}>Golden Answer:</div>
                           <div style={{ fontSize: '14px', whiteSpace: 'pre-wrap', borderLeft: '3px solid var(--color-success)', paddingLeft: '8px' }}>{c.ideal_answer}</div>
                         </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
