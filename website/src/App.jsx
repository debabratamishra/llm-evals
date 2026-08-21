import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Terminal, 
  Cpu, 
  Database, 
  Layers, 
  ShieldCheck, 
  Award, 
  BarChart3, 
  Check, 
  Copy, 
  ArrowRight, 
  Github, 
  Server, 
  ExternalLink,
  Info,
  Clock,
  Coins,
  RefreshCw,
  Sparkles,
  ChevronRight,
  Sun,
  Moon,
  Mail,
  Zap,
  CheckCircle2
} from 'lucide-react';

export default function App() {
  const [copiedText, setCopiedText] = useState('');
  const [activeTab, setActiveTab] = useState('judge');
  const [darkMode, setDarkMode] = useState(false); // Light is the default
  
  // Interactive Simulator States
  const [simDataset, setSimDataset] = useState('reasoning');
  const [simModelA, setSimModelA] = useState('anthropic/claude-sonnet-5');
  const [simModelB, setSimModelB] = useState('google/gemini-3.7-flash');
  const [simStatus, setSimStatus] = useState('idle'); // idle | running | done
  const [simProgress, setSimProgress] = useState(0);
  const [simLogs, setSimLogs] = useState([]);
  const [simResults, setSimResults] = useState(null);

  // Sync theme class with state
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedText(id);
    setTimeout(() => setCopiedText(''), 2000);
  };

  // Run simulation effect
  const runSimulation = () => {
    setSimStatus('running');
    setSimProgress(0);
    setSimLogs(['Initializing sandbox evaluator...', `Loading dataset: ${simDataset === 'reasoning' ? 'logical_reasoning_benchmark' : 'medical_qa_dataset'}`]);
    setSimResults(null);
  };

  useEffect(() => {
    if (simStatus !== 'running') return;

    const timer = setInterval(() => {
      setSimProgress((prev) => {
        const next = prev + 5;
        if (next === 20) {
          setSimLogs(logs => [...logs, `Connecting to OpenRouter gateway...`, `Model A: Loaded ${simModelA}`, `Model B: Loaded ${simModelB}`]);
        } else if (next === 45) {
          setSimLogs(logs => [...logs, `Evaluating Case 1/5: Running Exact Match & Similarity...`, `Evaluating Case 2/5: Evaluating Cost and Latency...`]);
        } else if (next === 70) {
          setSimLogs(logs => [...logs, `Evaluating Case 3/5: Querying LLM-as-a-Judge for Correctness...`, `Evaluating Case 4/5: Scoring Completeness & Clarity...`]);
        } else if (next === 90) {
          setSimLogs(logs => [...logs, `Evaluating Case 5/5: Compiling final leaderboard...`, `Finalizing run summary...`]);
        } else if (next >= 100) {
          clearInterval(timer);
          setSimStatus('done');
          
          const modelAName = simModelA.split('/').pop();
          const modelBName = simModelB.split('/').pop();
          const isModelABetter = (simModelA.includes('claude') || simModelA.includes('gpt')) && simDataset === 'reasoning';
          
          setSimResults({
            winner: isModelABetter ? modelAName : modelBName,
            metrics: {
              modelA: {
                name: modelAName,
                exactMatch: isModelABetter ? '92%' : '84%',
                similarity: isModelABetter ? '0.94' : '0.88',
                correctness: isModelABetter ? '4.95/5' : '4.62/5',
                completeness: isModelABetter ? '4.88/5' : '4.70/5',
                latency: isModelABetter ? '1.4s' : '1.1s',
                cost: '$0.0035'
              },
              modelB: {
                name: modelBName,
                exactMatch: isModelABetter ? '84%' : '95%',
                similarity: isModelABetter ? '0.87' : '0.96',
                correctness: isModelABetter ? '4.55/5' : '4.98/5',
                completeness: isModelABetter ? '4.65/5' : '4.91/5',
                latency: isModelABetter ? '0.8s' : '0.7s',
                cost: '$0.0022'
              }
            }
          });
          return 100;
        }
        return next;
      });
    }, 150);

    return () => clearInterval(timer);
  }, [simStatus, simDataset, simModelA, simModelB]);

  const quickStartCmds = [
    { id: 'clone', label: '1. Clone the repository', cmd: 'git clone https://github.com/debabratamishra/llm-evals.git\ncd llm-evals' },
    { id: 'start', label: '2. Launch local dev environment', cmd: 'chmod +x start_dashboard.sh\n./start_dashboard.sh' },
    { id: 'env', label: '3. Pre-load API credentials (Optional)', cmd: 'export OPENROUTER_API_KEY="your-key-here"\nexport NVIDIA_NIM_API_KEY="your-key-here"' }
  ];

  return (
    <div className="min-h-screen bg-bg-app text-text-primary font-sans transition-colors duration-300 relative overflow-hidden">
      {/* Background gradients */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-primary/10 opacity-[0.1] dark:opacity-[0.05] rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-1/3 right-1/4 w-[600px] h-[600px] bg-secondary/10 opacity-[0.1] dark:opacity-[0.03] rounded-full blur-[150px] pointer-events-none" />

      {/* Navigation */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-bg-app/80 border-b border-border-color">
        <div className="max-w-7xl mx-auto px-6 h-18 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-accent to-primary flex items-center justify-center shadow-lg shadow-primary/10">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-title font-extrabold text-xl tracking-tight bg-gradient-to-r from-text-primary to-text-secondary bg-clip-text text-transparent">
                LLM Evaluation Framework
              </span>
            </div>
          </div>
          
          <nav className="hidden lg:flex items-center gap-8 text-sm font-medium text-text-secondary">
            <a href="#features" className="hover:text-primary transition-colors">Features</a>
            <a href="#demo" className="hover:text-primary transition-colors">Interactive Demo</a>
            <a href="#architecture" className="hover:text-primary transition-colors">Architecture</a>
            <a href="#metrics" className="hover:text-primary transition-colors">Metrics</a>
            <a href="#enterprise" className="hover:text-primary transition-colors">Enterprise support</a>
            <a href="#quickstart" className="hover:text-primary transition-colors">Quick Start</a>
          </nav>

          <div className="flex items-center gap-4">
            {/* Theme Toggle Button */}
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className="p-2.5 rounded-xl bg-bg-sidebar hover:bg-bg-input border border-border-color transition-all text-text-secondary hover:text-text-primary"
              aria-label="Toggle theme"
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>

            <a 
              href="https://github.com/debabratamishra/llm-evals" 
              target="_blank" 
              rel="noreferrer" 
              className="p-2.5 rounded-xl bg-bg-sidebar hover:bg-bg-input border border-border-color transition-all text-text-secondary hover:text-text-primary"
              title="View on GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a 
              href="#quickstart" 
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-accent to-primary text-sm font-semibold text-white shadow-md hover:shadow-lg hover:-translate-y-0.5 transition-all"
            >
              Get Started
            </a>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative pt-24 pb-24 md:pt-32 md:pb-36 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="font-title font-black text-4xl sm:text-6xl md:text-7xl tracking-tight leading-tight max-w-5xl mx-auto">
            Benchmark, Evaluate & Compare{' '}
            <span className="bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
              LLMs Locally
            </span>
          </h1>
          
          <p className="mt-8 text-lg sm:text-xl text-text-secondary max-w-3xl mx-auto leading-relaxed">
            Upload datasets, run multi-provider evaluations, and compare metrics 
            side-by-side inside a beautiful real-time browser dashboard. Fully containerised, 
            lightweight, and zero key persistence.
          </p>
          
          <div className="mt-10 flex flex-wrap justify-center gap-4">
            <a 
              href="#demo" 
              className="px-8 py-4 rounded-xl bg-gradient-to-r from-accent to-primary font-semibold text-white shadow-lg hover:scale-105 transition-all flex items-center gap-2 group"
            >
              Run Interactive Demo
              <Play className="w-4 h-4 fill-current group-hover:translate-x-0.5 transition-transform" />
            </a>
            <a 
              href="#quickstart" 
              className="px-8 py-4 rounded-xl bg-bg-sidebar hover:bg-bg-input border border-border-color font-semibold text-text-secondary hover:text-text-primary transition-all flex items-center gap-2"
            >
              <Terminal className="w-4 h-4 text-text-muted" />
              Copy Startup Command
            </a>
          </div>

          {/* Glowing Mockup */}
          <div className="relative mt-20 max-w-5xl mx-auto rounded-2xl border border-border-color bg-bg-sidebar p-2 shadow-2xl">
            <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 via-transparent to-accent/5 rounded-2xl pointer-events-none" />
            <div className="rounded-xl overflow-hidden border border-border-color bg-bg-app">
              {/* Fake Window Header */}
              <div className="h-11 bg-bg-sidebar px-4 flex items-center justify-between border-b border-border-color">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-red-500/70" />
                  <span className="w-3 h-3 rounded-full bg-yellow-500/70" />
                  <span className="w-3 h-3 rounded-full bg-green-500/70" />
                </div>
                <div className="px-4 py-1 rounded bg-bg-input text-[11px] font-mono text-text-muted">
                  http://localhost:3000/arena
                </div>
                <div className="w-14" />
              </div>
              {/* Content mockup preview */}
              <div className="p-6 md:p-8 flex flex-col gap-6 text-left">
                <div className="flex items-center justify-between flex-wrap gap-4 border-b border-border-color pb-5">
                  <div>
                    <h4 className="text-lg font-bold text-text-primary">Arena Leaderboard: GPT vs Claude</h4>
                    <p className="text-xs text-text-muted">Dataset: reasoning_benchmark (25 cases)</p>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-green-500/10 border border-green-500/30 text-green-600 dark:text-green-400 text-xs font-semibold">
                    ✓ Evaluation Completed
                  </span>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Model A */}
                  <div className="p-5 rounded-xl bg-bg-sidebar/55 border border-accent/20 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-24 h-24 bg-accent/5 rounded-bl-full pointer-events-none" />
                    <span className="text-[10px] text-accent font-mono tracking-wider font-semibold uppercase">Model A</span>
                    <h5 className="font-bold text-lg mt-1 text-text-primary">gpt-5.6-terra</h5>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div>
                        <span className="text-xs text-text-muted block">LLM Correctness</span>
                        <span className="text-xl font-extrabold text-text-primary">4.92 / 5</span>
                      </div>
                      <div>
                        <span className="text-xs text-text-muted block">Average Latency</span>
                        <span className="text-xl font-extrabold text-text-primary">1.25s</span>
                      </div>
                      <div>
                        <span className="text-xs text-text-muted block">Exact Match</span>
                        <span className="text-xl font-extrabold text-text-primary">91%</span>
                      </div>
                      <div>
                        <span className="text-xs text-text-muted block">Cost (Est.)</span>
                        <span className="text-xl font-extrabold text-text-primary">$0.0240</span>
                      </div>
                    </div>
                  </div>

                  {/* Model B */}
                  <div className="p-5 rounded-xl bg-bg-sidebar/55 border border-primary/20 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-24 h-24 bg-primary/5 rounded-bl-full pointer-events-none" />
                    <span className="text-[10px] text-primary font-mono tracking-wider font-semibold uppercase">Model B (Winner)</span>
                    <h5 className="font-bold text-lg mt-1 text-text-primary">claude-sonnet-5</h5>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div>
                        <span className="text-xs text-text-muted block">LLM Correctness</span>
                        <span className="text-xl font-extrabold text-primary">4.96 / 5</span>
                      </div>
                      <div>
                        <span className="text-xs text-text-muted block">Average Latency</span>
                        <span className="text-xl font-extrabold text-primary">0.98s</span>
                      </div>
                      <div>
                        <span className="text-xs text-text-muted block">Exact Match</span>
                        <span className="text-xl font-extrabold text-primary">94%</span>
                      </div>
                      <div>
                        <span className="text-xs text-text-muted block">Cost (Est.)</span>
                        <span className="text-xl font-extrabold text-primary">$0.0180</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Capabilities Section */}
      <section id="features" className="py-24 border-t border-border-color bg-bg-sidebar/20 relative">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="font-title font-bold text-3xl sm:text-4xl text-text-primary">
              Framework Capabilities
            </h2>
            <p className="mt-4 text-text-secondary">
              Designed for developers who need to benchmark LLM outputs on specific datasets, 
              compare accuracy, and monitor API spend before pushing to production.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="p-8 rounded-2xl bg-bg-card border border-border-color hover:border-accent/30 transition-all hover:-translate-y-1 group shadow-sm">
              <div className="w-12 h-12 rounded-xl bg-accent/10 border border-accent/20 flex items-center justify-center text-accent group-hover:bg-accent group-hover:text-white transition-all">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold mt-6 text-text-primary">Multi-Backend Routing</h3>
              <p className="mt-3 text-sm text-text-secondary leading-relaxed">
                Connect to OpenRouter (200+ models), Nvidia NIM, or local endpoints via LiteLLM. Provide keys per evaluation run or preload env variables.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="p-8 rounded-2xl bg-bg-card border border-border-color hover:border-primary/30 transition-all hover:-translate-y-1 group shadow-sm">
              <div className="w-12 h-12 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-black dark:group-hover:text-white transition-all">
                <Database className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold mt-6 text-text-primary">Dataset Manager</h3>
              <p className="mt-3 text-sm text-text-secondary leading-relaxed">
                Upload CSV/JSON files or search and import directly from Hugging Face Hub. Built-in columns auto-detection maps your schema instantly.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="p-8 rounded-2xl bg-bg-card border border-border-color hover:border-pink-500/30 transition-all hover:-translate-y-1 group shadow-sm">
              <div className="w-12 h-12 rounded-xl bg-pink-500/10 border border-pink-500/20 flex items-center justify-center text-pink-500 group-hover:bg-pink-500 group-hover:text-white transition-all">
                <BarChart3 className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold mt-6 text-text-primary">Deep Cost & Latency Metrics</h3>
              <p className="mt-3 text-sm text-text-secondary leading-relaxed">
                Tracks token usage, response delays, and calculated prices in real-time. Know exactly how much every run costs you.
              </p>
            </div>

            {/* Feature 4 */}
            <div className="p-8 rounded-2xl bg-bg-card border border-border-color hover:border-green-500/30 transition-all hover:-translate-y-1 group shadow-sm">
              <div className="w-12 h-12 rounded-xl bg-green-500/10 border border-green-500/20 flex items-center justify-center text-green-400 group-hover:bg-green-500 group-hover:text-white transition-all">
                <Award className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold mt-6 text-text-primary">Arena Mode</h3>
              <p className="mt-3 text-sm text-text-secondary leading-relaxed">
                Set up head-to-head comparison battles. Run the same dataset through multiple models simultaneously and see which model takes the crown.
              </p>
            </div>

            {/* Feature 5 */}
            <div className="p-8 rounded-2xl bg-bg-card border border-border-color hover:border-yellow-500/30 transition-all hover:-translate-y-1 group shadow-sm">
              <div className="w-12 h-12 rounded-xl bg-yellow-500/10 border border-yellow-500/20 flex items-center justify-center text-yellow-400 group-hover:bg-yellow-500 group-hover:text-black transition-all">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold mt-6 text-text-primary">Secure Sandbox Testing</h3>
              <p className="mt-3 text-sm text-text-secondary leading-relaxed">
                Verify pipeline configurations locally with the Mock sandbox backend. Simulate errors and token limits without wasting real credits.
              </p>
            </div>

            {/* Feature 6 */}
            <div className="p-8 rounded-2xl bg-bg-card border border-border-color hover:border-blue-500/30 transition-all hover:-translate-y-1 group shadow-sm">
              <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-500 group-hover:bg-blue-600 group-hover:text-white transition-all">
                <Server className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold mt-6 text-text-primary">Single-Service Deploy</h3>
              <p className="mt-3 text-sm text-text-secondary leading-relaxed">
                Build frontend assets and serve them directly via FastAPI. Host the entire stack on Render free-tier with zero CORS configurations needed.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Arena Simulator (Showcase) */}
      <section id="demo" className="py-24 relative">
        <div className="max-w-4xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="font-title font-bold text-3xl text-text-primary">Interactive Evaluation Arena</h2>
            <p className="mt-3 text-text-secondary text-sm">
              Configure parameters below to simulate a real-time side-by-side evaluation.
            </p>
          </div>

          <div className="p-6 md:p-8 bg-bg-card border border-border-color rounded-2xl shadow-xl">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
              {/* Dataset Select */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold text-text-muted uppercase tracking-wider">Evaluation Dataset</label>
                <select 
                  value={simDataset} 
                  onChange={(e) => setSimDataset(e.target.value)}
                  disabled={simStatus === 'running'}
                  className="bg-bg-sidebar border border-border-color rounded-xl px-4 py-3 text-sm text-text-primary outline-none focus:border-primary disabled:opacity-50"
                >
                  <option value="reasoning">logical_reasoning_benchmark</option>
                  <option value="medical">medical_qa_dataset</option>
                </select>
              </div>

              {/* Model A Select */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold text-text-muted uppercase tracking-wider">Model A</label>
                <select 
                  value={simModelA} 
                  onChange={(e) => setSimModelA(e.target.value)}
                  disabled={simStatus === 'running'}
                  className="bg-bg-sidebar border border-border-color rounded-xl px-4 py-3 text-sm text-text-primary outline-none focus:border-accent disabled:opacity-50"
                >
                  <option value="anthropic/claude-sonnet-5">Claude Sonnet 5</option>
                  <option value="openai/gpt-5.6-terra">GPT-5.6 Terra</option>
                  <option value="nebula/muse-spark-1.2">Muse Spark 1.2</option>
                </select>
              </div>

              {/* Model B Select */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold text-text-muted uppercase tracking-wider">Model B</label>
                <select 
                  value={simModelB} 
                  onChange={(e) => setSimModelB(e.target.value)}
                  disabled={simStatus === 'running'}
                  className="bg-bg-sidebar border border-border-color rounded-xl px-4 py-3 text-sm text-text-primary outline-none focus:border-primary disabled:opacity-50"
                >
                  <option value="google/gemini-3.7-flash">Gemini 3.7 Flash</option>
                  <option value="google/gemini-3.1-pro-preview">Gemini 3.1 Pro Preview</option>
                </select>
              </div>
            </div>

            <div className="mt-8 flex justify-center">
              <button 
                onClick={runSimulation}
                disabled={simStatus === 'running'}
                className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-primary to-accent text-white font-extrabold text-sm shadow-md hover:scale-102 active:scale-98 transition-all flex items-center gap-2.5 cursor-pointer"
              >
                {simStatus === 'running' ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-white" />
                    Running Evaluation...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current text-white" />
                    Start Simulated Evaluation
                  </>
                )}
              </button>
            </div>

            {/* Run Progress & Logs */}
            {simStatus !== 'idle' && (
              <div className="mt-8 pt-6 border-t border-border-color">
                <div className="flex justify-between items-center text-xs font-mono text-text-muted mb-2">
                  <span>Pipeline Progress</span>
                  <span>{simProgress}%</span>
                </div>
                <div className="h-2 bg-bg-sidebar rounded-full overflow-hidden mb-4">
                  <div 
                    className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-150"
                    style={{ width: `${simProgress}%` }}
                  />
                </div>
                
                {/* Console Logs */}
                <div className="bg-bg-sidebar border border-border-color rounded-xl p-4 font-mono text-xs text-text-secondary h-32 overflow-y-auto flex flex-col gap-1.5">
                  {simLogs.map((log, idx) => (
                    <div key={idx} className="flex gap-2">
                      <span className="text-accent">&gt;</span>
                      <span>{log}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Simulated Results Leaderboard */}
            {simResults && (
              <div className="mt-8 pt-8 border-t border-border-color animate-fadeIn">
                <div className="flex items-center gap-3 mb-6">
                  <Award className="w-5 h-5 text-yellow-500" />
                  <h4 className="font-bold text-lg text-text-primary">Evaluation Battle Results</h4>
                  <span className="ml-auto text-xs px-2.5 py-1 bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 font-semibold border border-yellow-500/20 rounded-full flex items-center gap-1.5">
                    Winner: {simResults.winner}
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Model A Results Card */}
                  <div className={`p-5 rounded-xl border ${simResults.winner === simResults.metrics.modelA.name ? 'border-accent/30 bg-accent/5' : 'border-border-color bg-bg-sidebar/30'}`}>
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-bold font-title text-base text-text-primary">{simResults.metrics.modelA.name}</span>
                      {simResults.winner === simResults.metrics.modelA.name && (
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent/25 text-text-primary border border-accent/30 font-semibold">
                          Winner
                        </span>
                      )}
                    </div>
                    <div className="space-y-2.5 text-sm">
                      <div className="flex justify-between border-b border-border-color pb-1.5">
                        <span className="text-text-secondary">Exact Match</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelA.exactMatch}</span>
                      </div>
                      <div className="flex justify-between border-b border-border-color pb-1.5">
                        <span className="text-text-secondary">Sequence Similarity</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelA.similarity}</span>
                      </div>
                      <div className="flex justify-between border-b border-border-color pb-1.5">
                        <span className="text-text-secondary">LLM Correctness (Judge)</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelA.correctness}</span>
                      </div>
                      <div className="flex justify-between border-b border-border-color pb-1.5">
                        <span className="text-text-secondary">Average Latency</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelA.latency}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Est. Total Cost</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelA.cost}</span>
                      </div>
                    </div>
                  </div>

                  {/* Model B Results Card */}
                  <div className={`p-5 rounded-xl border ${simResults.winner === simResults.metrics.modelB.name ? 'border-primary/30 bg-primary/5' : 'border-border-color bg-bg-sidebar/30'}`}>
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-bold font-title text-base text-text-primary">{simResults.metrics.modelB.name}</span>
                      {simResults.winner === simResults.metrics.modelB.name && (
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-primary/25 text-text-primary border border-primary/30 font-semibold">
                          Winner
                        </span>
                      )}
                    </div>
                    <div className="space-y-2.5 text-sm">
                      <div className="flex justify-between border-b border-border-color pb-1.5">
                        <span className="text-text-secondary">Exact Match</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelB.exactMatch}</span>
                      </div>
                      <div className="flex justify-between border-b border-border-color pb-1.5">
                        <span className="text-text-secondary">Sequence Similarity</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelB.similarity}</span>
                      </div>
                      <div className="flex justify-between border-b border-border-color pb-1.5">
                        <span className="text-text-secondary">LLM Correctness (Judge)</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelB.correctness}</span>
                      </div>
                      <div className="flex justify-between border-b border-border-color pb-1.5">
                        <span className="text-text-secondary">Average Latency</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelB.latency}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Est. Total Cost</span>
                        <span className="font-semibold text-text-primary">{simResults.metrics.modelB.cost}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Architecture Flow Section */}
      <section id="architecture" className="py-24 border-t border-border-color bg-bg-sidebar/20 relative">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="font-title font-bold text-3xl text-text-primary">System Architecture</h2>
            <p className="mt-3 text-text-secondary text-sm">
              A modern single-origin design. The FastAPI backend handles the evaluation logic and hosts the React single page application.
            </p>
          </div>

          <div className="flex flex-col md:flex-row items-stretch justify-between gap-6 md:gap-4 relative">
            {/* Step 1: User / Browser */}
            <div className="flex-1 p-6 rounded-2xl bg-bg-card border border-border-color text-center flex flex-col items-center justify-between shadow-sm">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary mb-4">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <h4 className="font-bold text-base text-text-primary">Vite React SPA</h4>
                <p className="text-xs text-text-secondary mt-2">
                  Client-side dashboard displaying leaderboards, runs, and dataset forms.
                </p>
              </div>
              <div className="mt-4 px-3 py-1 rounded bg-bg-sidebar border border-border-color text-[10px] font-mono text-text-secondary">
                localhost:3000
              </div>
            </div>

            {/* Connection Arrow 1 */}
            <div className="flex items-center justify-center py-2 md:py-0">
              <ChevronRight className="w-6 h-6 text-text-muted rotate-90 md:rotate-0" />
            </div>

            {/* Step 2: FastAPI Service */}
            <div className="flex-1 p-6 rounded-2xl bg-bg-card border border-border-color text-center flex flex-col items-center justify-between shadow-sm">
              <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center text-green-600 dark:text-green-400 mb-4">
                <Server className="w-5 h-5" />
              </div>
              <div>
                <h4 className="font-bold text-base text-text-primary">FastAPI Backend</h4>
                <p className="text-xs text-text-secondary mt-2">
                  Handles upload, reads local JSON database files, routes API keys, and schedules evals.
                </p>
              </div>
              <div className="mt-4 px-3 py-1 rounded bg-bg-sidebar border border-border-color text-[10px] font-mono text-text-secondary">
                localhost:8000
              </div>
            </div>

            {/* Connection Arrow 2 */}
            <div className="flex items-center justify-center py-2 md:py-0">
              <ChevronRight className="w-6 h-6 text-text-muted rotate-90 md:rotate-0" />
            </div>

            {/* Step 3: LLM Providers */}
            <div className="flex-1 p-6 rounded-2xl bg-bg-card border border-border-color text-center flex flex-col items-center justify-between shadow-sm">
              <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center text-accent mb-4">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <h4 className="font-bold text-base text-text-primary">LiteLLM Gateway</h4>
                <p className="text-xs text-text-secondary mt-2">
                  Unified routing connection to OpenRouter, Nvidia NIM, or local mock sandbox.
                </p>
              </div>
              <div className="mt-4 px-3 py-1 rounded bg-bg-sidebar border border-border-color text-[10px] font-mono text-text-secondary">
                Multi-Provider Gateway
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Metrics matrix */}
      <section id="metrics" className="py-24 relative">
        <div className="max-w-4xl mx-auto px-6">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="font-title font-bold text-3xl text-text-primary">Comprehensive Evaluation Metrics</h2>
            <p className="mt-3 text-text-secondary text-sm">
              The framework supports both exact/fuzzy comparisons and semantic evaluations via LLM-as-a-judge.
            </p>
          </div>

          <div className="bg-bg-card border border-border-color rounded-2xl overflow-hidden shadow-xl">
            {/* Tabs */}
            <div className="flex border-b border-border-color">
              <button 
                onClick={() => setActiveTab('judge')}
                className={`flex-1 py-4 text-sm font-semibold tracking-wider transition-colors uppercase ${activeTab === 'judge' ? 'text-primary bg-bg-sidebar/30 border-b border-primary' : 'text-text-secondary hover:text-text-primary'}`}
              >
                LLM-as-a-Judge Rubric
              </button>
              <button 
                onClick={() => setActiveTab('deterministic')}
                className={`flex-1 py-4 text-sm font-semibold tracking-wider transition-colors uppercase ${activeTab === 'deterministic' ? 'text-primary bg-bg-sidebar/30 border-b border-primary' : 'text-text-secondary hover:text-text-primary'}`}
              >
                Deterministic Metrics
              </button>
            </div>

            {/* Tab contents */}
            <div className="p-6 md:p-8">
              {activeTab === 'judge' ? (
                <div className="space-y-6">
                  <div>
                    <h4 className="font-bold text-text-primary flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                      LLM Correctness (Factual Accuracy)
                    </h4>
                    <p className="text-xs text-text-secondary mt-1 pl-3.5 leading-relaxed">
                      Scored 1-5 by the judge model. Evaluates if the answer contains accurate facts that align with the reference solution, penalising hallucinations or factual errors.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-bold text-text-primary flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                      LLM Completeness (Coverage)
                    </h4>
                    <p className="text-xs text-text-secondary mt-1 pl-3.5 leading-relaxed">
                      Scored 1-5 by the judge model. Evaluates whether all parts of the question are answered, comparing information density against the ideal answer.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-bold text-text-primary flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-pink-400" />
                      LLM Clarity (Readability & Coherence)
                    </h4>
                    <p className="text-xs text-text-secondary mt-1 pl-3.5 leading-relaxed">
                      Scored 1-5 by the judge model. Assesses flow, syntax correctness, spelling, and structural logic of the generated markdown text.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <h4 className="font-bold text-text-primary flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                      Exact Match (EM)
                    </h4>
                    <p className="text-xs text-text-secondary mt-1 pl-3.5 leading-relaxed">
                      Returns a binary 1 or 0 score. Text outputs are normalised (converting lowercase, stripping whitespace, removing punctuation) to check if the generated output matches exactly.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-bold text-text-primary flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-yellow-500" />
                      Sequence Similarity
                    </h4>
                    <p className="text-xs text-text-secondary mt-1 pl-3.5 leading-relaxed">
                      Floating-point ratio [0.0 - 1.0]. Calculates the Levenshtein-based sequence overlap score representing literal similarity, regardless of word meanings.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-bold text-text-primary flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                      Performance (Latency & Cost)
                    </h4>
                    <p className="text-xs text-text-secondary mt-1 pl-3.5 leading-relaxed">
                      Calculates the exact response execution time per test case (in seconds) and multiplies input/output token counts with provider pricing schemas to track absolute spending in USD.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Enterprise Offerings & Support Section */}
      <section id="enterprise" className="py-24 border-t border-border-color relative">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="font-title font-bold text-3xl text-text-primary">Enterprise Support & Custom Offerings</h2>
            <p className="mt-3 text-text-secondary text-sm">
              We offer technical services, SLAs, and custom feature implementations to scale your LLM evaluation pipelines.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            {/* Offering 1 */}
            <div className="p-6 rounded-2xl bg-bg-card border border-border-color shadow-sm flex flex-col justify-between">
              <div>
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary mb-4">
                  <Zap className="w-5 h-5" />
                </div>
                <h4 className="font-bold text-lg text-text-primary mb-3">Custom Evals & Integrations</h4>
                <p className="text-sm text-text-secondary leading-relaxed">
                  Design and implementation of proprietary evaluation metrics, judge rubrics, and connectors for internal dataset formats (databases, AWS S3, or custom APIs).
                </p>
              </div>
              <ul className="mt-6 space-y-2 text-xs text-text-secondary">
                <li className="flex items-center gap-2"><CheckCircle2 className="w-3.5 h-3.5 text-primary" /> Proprietary judge models</li>
                <li className="flex items-center gap-2"><CheckCircle2 className="w-3.5 h-3.5 text-primary" /> Data pipelines (ETL)</li>
              </ul>
            </div>

            {/* Offering 2 */}
            <div className="p-6 rounded-2xl bg-bg-card border border-border-color shadow-sm flex flex-col justify-between">
              <div>
                <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center text-accent mb-4">
                  <Server className="w-5 h-5" />
                </div>
                <h4 className="font-bold text-lg text-text-primary mb-3">VPC & On-Prem Deployment</h4>
                <p className="text-sm text-text-secondary leading-relaxed">
                  Enterprise-grade isolated deployment guidance in AWS VPC, Azure, or private hardware. Setup SSO/SAML auth and security-hardened API proxy controls.
                </p>
              </div>
              <ul className="mt-6 space-y-2 text-xs text-text-secondary">
                <li className="flex items-center gap-2"><CheckCircle2 className="w-3.5 h-3.5 text-accent" /> AWS, Azure, & GCP VPC</li>
                <li className="flex items-center gap-2"><CheckCircle2 className="w-3.5 h-3.5 text-accent" /> SAML / OAuth Integrations</li>
              </ul>
            </div>

            {/* Offering 3 */}
            <div className="p-6 rounded-2xl bg-bg-card border border-border-color shadow-sm flex flex-col justify-between">
              <div>
                <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center text-green-600 dark:text-green-400 mb-4">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <h4 className="font-bold text-lg text-text-primary mb-3">SLA & Priority Assistance</h4>
                <p className="text-sm text-text-secondary leading-relaxed">
                  Guaranteed response times, priority bug patching, dedicated engineering support channels, and regular maintenance check-ups.
                </p>
              </div>
              <ul className="mt-6 space-y-2 text-xs text-text-secondary">
                <li className="flex items-center gap-2"><CheckCircle2 className="w-3.5 h-3.5 text-green-600 dark:text-green-400" /> Dedicated support SLA</li>
                <li className="flex items-center gap-2"><CheckCircle2 className="w-3.5 h-3.5 text-green-600 dark:text-green-400" /> Priority hotfixes</li>
              </ul>
            </div>
          </div>

          {/* Contact CTA */}
          <div className="max-w-2xl mx-auto p-8 rounded-2xl bg-gradient-to-r from-accent/5 to-primary/5 border border-primary/20 text-center flex flex-col items-center gap-6">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center text-primary shadow-inner">
              <Mail className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-xl text-text-primary mb-2">Discuss Enterprise Offerings</h4>
              <p className="text-sm text-text-secondary max-w-lg leading-relaxed">
                Reach out to discuss custom features, SLA requirements, or deployment integrations for your organization.
              </p>
            </div>
            <a 
              href="mailto:debabrata.mishra641@gmail.com" 
              className="px-8 py-3 rounded-xl bg-bg-sidebar hover:bg-bg-input border border-border-color hover:border-primary/50 text-text-primary font-bold text-sm flex items-center gap-2 hover:-translate-y-0.5 transition-all shadow-sm cursor-pointer"
            >
              Contact Support
              <ArrowRight className="w-4 h-4 text-primary" />
            </a>
          </div>
        </div>
      </section>

      {/* Quick Start Guide Section */}
      <section id="quickstart" className="py-24 border-t border-border-color bg-bg-sidebar/20 relative">
        <div className="max-w-4xl mx-auto px-6">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="font-title font-bold text-3xl text-text-primary">One-Command Quick Start</h2>
            <p className="mt-3 text-text-secondary text-sm">
              Spin up the local developer server using uv and npm in less than a minute.
            </p>
          </div>

          <div className="space-y-6">
            {quickStartCmds.map((item) => (
              <div key={item.id} className="flex flex-col gap-2">
                <span className="text-sm font-semibold text-text-secondary">{item.label}</span>
                <div className="relative group">
                  <pre className="bg-bg-sidebar border border-border-color rounded-xl p-5 overflow-x-auto font-mono text-sm text-primary pr-12">
                    <code>{item.cmd}</code>
                  </pre>
                  <button 
                    onClick={() => copyToClipboard(item.cmd, item.id)}
                    className="absolute top-4 right-4 p-2 rounded-lg bg-bg-card hover:bg-bg-input border border-border-color group-hover:opacity-100 opacity-60 hover:opacity-100 transition-all text-text-muted hover:text-text-primary cursor-pointer"
                    title="Copy command"
                  >
                    {copiedText === item.id ? (
                      <Check className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* Hosting Guide alert */}
          <div className="mt-10 p-5 rounded-2xl bg-primary/5 border border-primary/10 flex gap-4">
            <Info className="w-5 h-5 text-primary shrink-0 mt-0.5" />
            <div className="text-xs leading-relaxed text-text-secondary">
              <span className="font-semibold text-text-primary block mb-1">Production Deployment</span>
              Ready to host your dashboard on a server? Check out <a href="https://github.com/debabratamishra/llm-evals/blob/main/DEPLOY.md" target="_blank" rel="noreferrer" className="text-primary underline hover:text-primary/80 inline-flex items-center gap-0.5 font-semibold">DEPLOY.md<ExternalLink className="w-3 h-3" /></a>. The backend builds and mounts the frontend into the Python app, letting you deploy the entire dashboard on a single Render free web service.
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-border-color bg-bg-sidebar/10">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6 text-sm text-text-muted">
          <div className="flex items-center gap-2">
            <span className="font-title font-semibold text-text-secondary">LLM Evaluation Framework</span>
            <span>•</span>
            <span>Released under Apache-2.0 License</span>
          </div>
          <div className="flex items-center gap-6">
            <a href="https://github.com/debabratamishra/llm-evals" target="_blank" rel="noreferrer" className="hover:text-text-primary transition-colors flex items-center gap-1.5">
              <Github className="w-4 h-4" />
              GitHub
            </a>
            <a href="https://github.com/debabratamishra/llm-evals/blob/main/LICENSE" target="_blank" rel="noreferrer" className="hover:text-text-primary transition-colors">
              License
            </a>
            <a href="https://github.com/debabratamishra/llm-evals/blob/main/README.md" target="_blank" rel="noreferrer" className="hover:text-text-primary transition-colors">
              Documentation
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
