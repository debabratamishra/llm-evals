import React, { useState, useEffect } from 'react';
import { LayoutDashboard, Database, PlayCircle, History, Sparkles, AlertCircle, CheckCircle, RefreshCcw } from 'lucide-react';
import DashboardOverview from './components/DashboardOverview';
import DatasetManager from './components/DatasetManager';
import EvaluationRunner from './components/EvaluationRunner';
import RunsHistory from './components/RunsHistory';
import RunDetails from './components/RunDetails';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview'); // overview, datasets, runner, history, details
  const [selectedRunId, setSelectedRunId] = useState(null);
  
  const [datasets, setDatasets] = useState([]);
  const [runs, setRuns] = useState([]);
  const [apiKeysSet, setApiKeysSet] = useState({
    nvidia_nim_api_key_set: false,
    nvidia_nim_base_url_set: false,
    nvidia_nim_base_url: 'https://integrate.api.nvidia.com/v1',
    openrouter_api_key_set: false,
  });
  
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState(null);

  // Fetch initial data
  const fetchData = async () => {
    try {
      const [datasetsRes, runsRes, keysRes] = await Promise.all([
        fetch('/api/datasets'),
        fetch('/api/runs'),
        fetch('/api/check-keys')
      ]);

      if (datasetsRes.ok) setDatasets(await datasetsRes.json());
      if (runsRes.ok) setRuns(await runsRes.json());
      if (keysRes.ok) setApiKeysSet(await keysRes.json());
    } catch (err) {
      console.error("Error fetching initial dashboard data: ", err);
      showToast('error', 'Could not establish connection to the FastAPI backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Toast notifier helper
  const showToast = (type, message) => {
    setToast({ type, message });
  };

  // Auto clear toasts after 4 seconds
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  const handleViewRun = (runId) => {
    setSelectedRunId(runId);
    setActiveTab('details');
  };

  const handleRunComplete = async (newRunId) => {
    await fetchData(); // Refresh data
    setSelectedRunId(newRunId);
    setActiveTab('details');
  };

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'overview':
        return <DashboardOverview runs={runs} onViewRun={handleViewRun} />;
      case 'datasets':
        return <DatasetManager datasets={datasets} onRefresh={fetchData} setToast={showToast} />;
      case 'runner':
        return (
          <EvaluationRunner 
            datasets={datasets} 
            apiKeysSet={apiKeysSet} 
            onRunComplete={handleRunComplete} 
            setToast={showToast} 
          />
        );
      case 'history':
        return (
          <RunsHistory 
            runs={runs} 
            onViewRun={handleViewRun} 
            onRefresh={fetchData} 
            setToast={showToast} 
          />
        );
      case 'details':
        return (
          <RunDetails 
            runId={selectedRunId} 
            onBack={() => { setActiveTab('history'); setSelectedRunId(null); }} 
            setToast={showToast} 
          />
        );
      default:
        return <DashboardOverview runs={runs} onViewRun={handleViewRun} />;
    }
  };

  if (loading) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        height: '100vh', backgroundColor: '#080a10', color: '#f3f4f6'
      }}>
        <div style={{
          width: '40px', height: '40px', border: '3px solid rgba(0, 242, 254, 0.1)',
          borderTopColor: '#00f2fe', borderRadius: '50%', animation: 'spin 1s linear infinite',
          marginBottom: '16px'
        }} />
        <span style={{ fontFamily: 'Outfit', fontWeight: 600, letterSpacing: '0.05em' }}>BOOTING EVAL SYSTEM</span>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="logo-container">
          <div className="logo-icon">𝝙</div>
          <span className="logo-text">LLM Evals</span>
        </div>

        <ul className="nav-links">
          <li 
            className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <LayoutDashboard className="nav-icon" /> Dashboard
          </li>
          <li 
            className={`nav-item ${activeTab === 'datasets' ? 'active' : ''}`}
            onClick={() => setActiveTab('datasets')}
          >
            <Database className="nav-icon" /> Dataset Manager
          </li>
          <li 
            className={`nav-item ${activeTab === 'runner' ? 'active' : ''}`}
            onClick={() => setActiveTab('runner')}
          >
            <PlayCircle className="nav-icon" /> Run Eval
          </li>
          <li 
            className={`nav-item ${activeTab === 'history' || activeTab === 'details' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History className="nav-icon" /> Eval History
          </li>
        </ul>


      </aside>

      {/* Main Panel Viewport */}
      <main className="main-content">
        {renderActiveTab()}
      </main>

      {/* Toast Alert Notifications */}
      {toast && (
        <div className="toast-container">
          <div className={`toast ${toast.type === 'success' ? 'toast-success' : 'toast-error'}`}>
            {toast.type === 'success' ? (
              <CheckCircle size={18} style={{ color: 'var(--color-success)' }} />
            ) : (
              <AlertCircle size={18} style={{ color: 'var(--color-danger)' }} />
            )}
            <span className="toast-message">{toast.message}</span>
          </div>
        </div>
      )}
    </div>
  );
}
