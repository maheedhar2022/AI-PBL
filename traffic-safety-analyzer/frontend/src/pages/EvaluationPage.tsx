import { useEffect, useState, useCallback } from 'react'
import {
  BarChart2, RefreshCw, Play, CheckCircle, AlertCircle, Loader, Trash2
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { runEvaluation, listEvalResults, listDatasets, getEvalStatus, deleteEvalResult } from '../services/api'

interface EvalResult {
  id: string
  experiment_name: string
  model_name: string
  dataset_name: string
  run_timestamp: string
  metrics: {
    precision?: number
    recall?: number
    f1?: number
    map50?: number
    map50_95?: number
    avg_inference_ms?: number
    fps?: number
    total_images?: number
    status?: string
    error_message?: string
    per_class?: Record<string, { precision: number; recall: number; f1: number }>
  }
}

export default function EvaluationPage() {
  const [results, setResults] = useState<EvalResult[]>([])
  const [datasets, setDatasets] = useState<{ name: string; path: string; image_count_val: number }[]>([])
  const [running, setRunning] = useState(false)
  const [form, setForm] = useState({
    dataset_path: '',
    dataset_name: '',
    iou_threshold: '0.50',
    confidence_threshold: '0.25',
    experiment_name: '',
    notes: '',
  })
  const [message, setMessage] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const [r, d] = await Promise.all([listEvalResults(), listDatasets()])
      setResults(r)
      setDatasets(d.datasets || [])
    } catch {}
  }, [])

  const checkStatus = useCallback(async () => {
    try {
      const st = await getEvalStatus()
      setRunning(st.running)
      if (!st.running) load()
    } catch {}
  }, [load])

  useEffect(() => {
    load()
    const timer = setInterval(checkStatus, 5000)
    return () => clearInterval(timer)
  }, [load, checkStatus])

  const handleRun = async () => {
    if (!form.dataset_path || !form.dataset_name) {
      setMessage('Dataset path and name are required.')
      return
    }
    setMessage(null)
    try {
      await runEvaluation({
        dataset_path: form.dataset_path,
        dataset_name: form.dataset_name,
        iou_threshold: parseFloat(form.iou_threshold),
        confidence_threshold: parseFloat(form.confidence_threshold),
        experiment_name: form.experiment_name,
        notes: form.notes,
      })
      setRunning(true)
      setMessage('Evaluation started. Results will appear when complete.')
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setMessage(err?.response?.data?.detail ?? 'Failed to start evaluation.')
    }
  }

  const handleDeleteResult = async (id: string) => {
    if (!confirm('Delete this evaluation record?')) return
    try {
      await deleteEvalResult(id)
      setResults(prev => prev.filter(r => r.id !== id))
    } catch { /* ignore */ }
  }

  const latest = results[0]
  const chartData = results.slice(0, 10).reverse().map(r => ({
    name: r.experiment_name.slice(0, 16) || r.dataset_name.slice(0, 16),
    Precision: +(r.metrics.precision || 0).toFixed(3),
    Recall: +(r.metrics.recall || 0).toFixed(3),
    F1: +(r.metrics.f1 || 0).toFixed(3),
    mAP50: +(r.metrics.map50 || 0).toFixed(3),
  }))

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">AI Evaluation</h1>
          <p className="text-gray-400 text-sm">Model performance metrics and experiment history</p>
        </div>
        <button onClick={load} className="btn-secondary flex items-center gap-2 text-sm">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {/* Top Metrics */}
      {latest && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <MetricCard label="Precision" value={latest.metrics.precision} />
          <MetricCard label="Recall" value={latest.metrics.recall} />
          <MetricCard label="F1 Score" value={latest.metrics.f1} />
          <MetricCard label="mAP@50" value={latest.metrics.map50} />
          <MetricCard label="mAP@50-95" value={latest.metrics.map50_95} />
        </div>
      )}

      {!latest && (
        <div className="card text-gray-500 text-sm text-center py-8">
          <BarChart2 className="w-10 h-10 mx-auto mb-2 opacity-30" />
          <div className="font-medium">Not evaluated</div>
          <div className="text-xs mt-1">Run an evaluation below to see performance metrics.</div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Chart */}
        {chartData.length > 0 && (
          <div className="card">
            <h2 className="font-semibold text-white mb-3">Experiment History</h2>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                  labelStyle={{ color: '#f3f4f6' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Precision" fill="#3b82f6" />
                <Bar dataKey="Recall" fill="#10b981" />
                <Bar dataKey="F1" fill="#f59e0b" />
                <Bar dataKey="mAP50" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Run Form */}
        <div className="card space-y-3">
          <h2 className="font-semibold text-white flex items-center gap-2">
            <Play className="w-4 h-4 text-green-400" />
            Run Evaluation
          </h2>

          {/* Available datasets */}
          {datasets.length > 0 && (
            <div>
              <div className="text-xs text-gray-500 mb-1">Available datasets:</div>
              {datasets.map(d => (
                <button
                  key={d.path}
                  onClick={() => setForm(f => ({ ...f, dataset_path: d.path, dataset_name: d.name }))}
                  className="w-full text-left bg-gray-800 hover:bg-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 mb-1 transition-colors"
                >
                  <b>{d.name}</b>
                  <span className="text-gray-500 text-xs ml-2">{d.image_count_val} val images</span>
                </button>
              ))}
            </div>
          )}

          <div className="space-y-2">
            <input
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500"
              placeholder="Dataset path (e.g. ./research/datasets/my_dataset)"
              value={form.dataset_path}
              onChange={e => setForm(f => ({ ...f, dataset_path: e.target.value }))}
            />
            <input
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500"
              placeholder="Dataset name"
              value={form.dataset_name}
              onChange={e => setForm(f => ({ ...f, dataset_name: e.target.value }))}
            />
            <div className="flex gap-2">
              <div className="flex-1">
                <label className="text-xs text-gray-500">IoU Threshold</label>
                <input
                  type="number" step="0.05" min="0.1" max="0.95"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                  value={form.iou_threshold}
                  onChange={e => setForm(f => ({ ...f, iou_threshold: e.target.value }))}
                />
              </div>
              <div className="flex-1">
                <label className="text-xs text-gray-500">Confidence</label>
                <input
                  type="number" step="0.05" min="0.1" max="0.95"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                  value={form.confidence_threshold}
                  onChange={e => setForm(f => ({ ...f, confidence_threshold: e.target.value }))}
                />
              </div>
            </div>
            <input
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500"
              placeholder="Experiment name (optional)"
              value={form.experiment_name}
              onChange={e => setForm(f => ({ ...f, experiment_name: e.target.value }))}
            />
          </div>

          {message && (
            <div className="text-xs text-blue-300 bg-blue-900/30 rounded px-3 py-2">{message}</div>
          )}

          <button
            onClick={handleRun}
            disabled={running}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {running ? <><Loader className="w-4 h-4 animate-spin" /> Running…</> : <><Play className="w-4 h-4" /> Start Evaluation</>}
          </button>
        </div>
      </div>

      {/* Results table */}
      {results.length > 0 && (
        <div className="card overflow-x-auto">
          <h2 className="font-semibold text-white mb-3">Experiment History</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs border-b border-gray-800">
                <th className="text-left py-2">Experiment</th>
                <th className="text-left py-2">Model</th>
                <th className="text-left py-2">Dataset</th>
                <th className="text-right py-2">P</th>
                <th className="text-right py-2">R</th>
                <th className="text-right py-2">F1</th>
                <th className="text-right py-2">mAP50</th>
                <th className="text-right py-2">FPS</th>
                <th className="text-left py-2">Status</th>
                <th className="py-2 w-8"></th>
              </tr>
            </thead>
            <tbody>
              {results.map(r => (
                <tr key={r.id} className="border-b border-gray-800 hover:bg-gray-800/40 transition-colors">
                  <td className="py-2 text-gray-300 text-xs truncate max-w-[120px]">{r.experiment_name}</td>
                  <td className="py-2 text-gray-400 text-xs">{r.model_name}</td>
                  <td className="py-2 text-gray-400 text-xs">{r.dataset_name}</td>
                  <td className="py-2 text-right text-gray-300 font-mono text-xs">
                    {r.metrics.status === 'error' ? '—' : (r.metrics.precision ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2 text-right text-gray-300 font-mono text-xs">
                    {r.metrics.status === 'error' ? '—' : (r.metrics.recall ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2 text-right text-gray-300 font-mono text-xs">
                    {r.metrics.status === 'error' ? '—' : (r.metrics.f1 ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2 text-right text-gray-300 font-mono text-xs">
                    {r.metrics.status === 'error' ? '—' : (r.metrics.map50 ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2 text-right text-gray-300 font-mono text-xs">
                    {r.metrics.fps?.toFixed(1) ?? '—'}
                  </td>
                  <td className="py-2 text-xs">
                    {r.metrics.status === 'error'
                      ? <span className="text-red-400 flex items-center gap-1"><AlertCircle className="w-3 h-3" />Error</span>
                      : <span className="text-green-400 flex items-center gap-1"><CheckCircle className="w-3 h-3" />Done</span>
                    }
                  </td>
                  <td className="py-2 text-right">
                    <button
                      onClick={() => handleDeleteResult(r.id)}
                      className="text-gray-600 hover:text-red-400 p-1 transition-colors"
                      title="Delete result"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value?: number }) {
  const display = value !== undefined ? (value * 100).toFixed(1) + '%' : 'N/A'
  const color = !value ? 'text-gray-600' :
    value >= 0.7 ? 'text-green-400' :
    value >= 0.4 ? 'text-yellow-400' : 'text-red-400'
  return (
    <div className="card text-center">
      <div className="text-gray-500 text-xs mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>{display}</div>
    </div>
  )
}

