import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AlertTriangle, Video, Clock, CheckSquare,
  Activity, RefreshCw, BarChart3, PieChart as PieIcon, Shield, Settings, ArrowRight
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, PieChart, Pie, Cell
} from 'recharts'
import { getDashboardStats, getRecentIncidents, getDashboardAnalytics } from '../services/api'
import { formatDistanceToNow } from 'date-fns'

interface Stats {
  total_incidents: number
  helmet_incidents: number
  triple_riding_incidents: number
  pending_review: number
  camera_running: boolean
  pipeline_running: boolean
  pipeline_metrics: {
    frames_processed?: number
    inference_fps?: number
    last_inference_ms?: number
    incidents_created?: number
    active_vehicles?: number
    traffic_density?: string
    vehicle_counts?: Record<string, number>
  }
}

interface RecentIncident {
  id: string
  violation_type: string
  event_timestamp: string
  confidence: number
  review_status: string
  outcome: string
}

interface AnalyticsData {
  violation_distribution: { name: string; value: number; color: string }[]
  review_distribution: { name: string; value: number; color: string }[]
  timeline: { time: string; helmet: number; triple: number }[]
  traffic_metrics: Record<string, unknown>
}

export default function Overview() {
  const navigate = useNavigate()
  const [stats, setStats] = useState<Stats | null>(null)
  const [recent, setRecent] = useState<RecentIncident[]>([])
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const [s, r, a] = await Promise.all([
        getDashboardStats(),
        getRecentIncidents(8),
        getDashboardAnalytics(),
      ])
      setStats(s)
      setRecent(r)
      setAnalytics(a)
      setError(null)
    } catch {
      setError('Cannot reach API. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
    if (!autoRefresh) return
    const timer = setInterval(load, 8000)
    return () => clearInterval(timer)
  }, [load, autoRefresh])

  const violationBadge = (type: string) =>
    type === 'helmet_violation'
      ? <span className="badge-helmet">Helmet</span>
      : <span className="badge badge-triple">Triple Riding</span>

  const hasTimeline = analytics && analytics.timeline && analytics.timeline.length > 0
  const hasReviews = analytics && analytics.review_distribution.some(r => r.value > 0)

  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Overview & Traffic Safety Analytics</h1>
          <p className="text-gray-400 text-sm mt-0.5">Live monitoring telemetry, incident distribution, and violation audit trail</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAutoRefresh(r => !r)}
            className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
              autoRefresh ? 'bg-green-950/40 border-green-800 text-green-300' : 'bg-gray-800 border-gray-750 text-gray-400'
            }`}
          >
            {autoRefresh ? '● Live Refreshing (8s)' : 'Paused'}
          </button>
          <button onClick={load} className="btn-secondary flex items-center gap-1.5 text-xs py-1.5 px-3">
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Interactive Clickable Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<AlertTriangle className="w-5 h-5 text-orange-400" />}
          label="Total Incidents"
          value={stats?.total_incidents ?? '—'}
          sub="recorded events"
          onClick={() => navigate('/incidents')}
        />
        <StatCard
          icon={<Shield className="w-5 h-5 text-red-400" />}
          label="Helmet Violations"
          value={stats?.helmet_incidents ?? '—'}
          sub="potential violations"
          onClick={() => navigate('/incidents?type=helmet_violation')}
        />
        <StatCard
          icon={<Video className="w-5 h-5 text-purple-400" />}
          label="Triple Riding"
          value={stats?.triple_riding_incidents ?? '—'}
          sub="potential violations"
          onClick={() => navigate('/incidents?type=triple_riding')}
        />
        <StatCard
          icon={<CheckSquare className="w-5 h-5 text-yellow-400" />}
          label="Pending Review"
          value={stats?.pending_review ?? '—'}
          sub="awaiting review"
          onClick={() => navigate('/incidents?status=pending')}
          highlight={stats ? stats.pending_review > 0 : false}
        />
      </div>

      {/* Quick Action Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-gray-900 border border-gray-800 p-3 rounded-xl text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-gray-400 font-medium">Quick Navigation:</span>
          <button
            onClick={() => navigate('/monitoring')}
            className={`px-3 py-1.5 rounded-lg border font-medium flex items-center gap-1.5 transition-colors ${
              stats?.pipeline_running
                ? 'bg-green-600/20 border-green-500/50 text-green-300 hover:bg-green-600/30'
                : 'bg-blue-600/20 border-blue-500/50 text-blue-300 hover:bg-blue-600/30'
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            {stats?.pipeline_running ? 'Go to Active Feed ●' : 'Launch CCTV Monitoring'}
          </button>
          <button
            onClick={() => navigate('/incidents?status=pending')}
            className="px-3 py-1.5 rounded-lg bg-yellow-600/20 border border-yellow-500/50 text-yellow-300 hover:bg-yellow-600/30 font-medium flex items-center gap-1.5 transition-colors"
          >
            <CheckSquare className="w-3.5 h-3.5 text-yellow-400" />
            Review Queue ({stats?.pending_review ?? 0})
          </button>
          <button
            onClick={() => navigate('/settings')}
            className="px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-750 text-gray-300 hover:bg-gray-750 font-medium flex items-center gap-1.5 transition-colors"
          >
            <Settings className="w-3.5 h-3.5 text-purple-400" /> Adjust Sensitivity
          </button>
        </div>
        <div className="text-xs text-gray-500">
          Tip: Click any stat card above to filter incidents directly.
        </div>
      </div>

      {/* Analytics Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Timeline Chart */}
        <div className="card lg:col-span-2">
          <h2 className="font-semibold text-white mb-3 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-blue-400" />
            Violation Incidents Activity
          </h2>
          {hasTimeline ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={analytics?.timeline}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                  labelStyle={{ color: '#f3f4f6' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="helmet" name="Helmet" fill="#ef4444" radius={[4, 4, 0, 0]} />
                <Bar dataKey="triple" name="Triple Riding" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-44 flex flex-col items-center justify-center text-gray-500 text-xs">
              <BarChart3 className="w-8 h-8 opacity-30 mb-2" />
              <span>No timeline activity recorded yet. Run live monitoring to log events.</span>
            </div>
          )}
        </div>

        {/* Breakdown Donut / Pie */}
        <div className="card">
          <h2 className="font-semibold text-white mb-3 flex items-center gap-2">
            <PieIcon className="w-4 h-4 text-emerald-400" />
            Review Status Distribution
          </h2>
          {hasReviews ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={analytics?.review_distribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {analytics?.review_distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                  labelStyle={{ color: '#f3f4f6' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-44 flex flex-col items-center justify-center text-gray-500 text-xs">
              <PieIcon className="w-8 h-8 opacity-30 mb-2" />
              <span>No reviewed incidents yet.</span>
            </div>
          )}
        </div>
      </div>

      {/* System Status & Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card">
          <h2 className="font-semibold text-white mb-3 flex items-center gap-2">
            <Clock className="w-4 h-4 text-blue-400" />
            System Status & Telemetry
          </h2>
          <div className="space-y-3">
            <StatusRow label="Camera Feed" active={stats?.camera_running ?? false} />
            <StatusRow label="AI Detection & Tracking Pipeline" active={stats?.pipeline_running ?? false} />
            <div className="border-t border-gray-800 pt-3 space-y-1.5">
              <MetricRow label="Frames Processed" value={stats?.pipeline_metrics?.frames_processed?.toLocaleString() ?? '0'} />
              <MetricRow label="Inference FPS" value={`${stats?.pipeline_metrics?.inference_fps?.toFixed(1) ?? '0'} fps`} />
              <MetricRow label="Inference Latency" value={`${stats?.pipeline_metrics?.last_inference_ms?.toFixed(0) ?? '0'} ms`} />
              <MetricRow label="Active Road Vehicles" value={`${stats?.pipeline_metrics?.active_vehicles ?? 0}`} />
              <MetricRow label="Traffic Flow Density" value={stats?.pipeline_metrics?.traffic_density ?? 'Light'} />
              <MetricRow label="Total Incidents Created" value={`${stats?.pipeline_metrics?.incidents_created ?? 0}`} />
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              Recent Incidents Audit Log
            </h2>
            <button
              onClick={() => navigate('/incidents')}
              className="text-xs text-blue-400 hover:underline flex items-center gap-1"
            >
              View all <ArrowRight className="w-3 h-3" />
            </button>
          </div>
          {loading ? (
            <div className="text-gray-500 text-sm">Loading…</div>
          ) : recent.length === 0 ? (
            <div className="text-gray-500 text-sm">No incidents recorded yet.</div>
          ) : (
            <div className="space-y-2">
              {recent.map(inc => (
                <div
                  key={inc.id}
                  onClick={() => navigate(`/incidents`)}
                  className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0 hover:bg-gray-800/40 rounded px-2 cursor-pointer transition-colors"
                >
                  <div className="flex items-center gap-2">
                    {violationBadge(inc.violation_type)}
                    <span className="text-gray-400 text-xs">
                      {formatDistanceToNow(new Date(inc.event_timestamp), { addSuffix: true })}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500 font-mono">{(inc.confidence * 100).toFixed(0)}%</span>
                    <span className={`badge text-xs ${inc.review_status === 'pending' ? 'badge-pending' : inc.review_status === 'confirmed' ? 'badge-confirmed' : 'bg-gray-800 text-gray-400'}`}>
                      {inc.review_status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon, label, value, sub, onClick, highlight }: {
  icon: React.ReactNode
  label: string
  value: number | string
  sub: string
  onClick?: () => void
  highlight?: boolean
}) {
  return (
    <div
      onClick={onClick}
      className={`stat-card transition-all ${
        onClick ? 'cursor-pointer hover:border-blue-600/70 hover:bg-gray-850 hover:shadow-lg' : ''
      } ${highlight ? 'border-yellow-600/40' : ''}`}
    >
      <div className="flex items-center gap-2 text-gray-400 text-sm">
        {icon}
        <span>{label}</span>
      </div>
      <div className="text-3xl font-bold text-white">{value}</div>
      <div className="text-gray-500 text-xs flex items-center justify-between mt-1">
        <span>{sub}</span>
        {onClick && <span className="text-[10px] text-blue-400 font-medium">View →</span>}
      </div>
    </div>
  )
}

function StatusRow({ label, active }: { label: string; active: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-400 text-sm">{label}</span>
      <span className={`flex items-center gap-1.5 text-xs font-medium ${active ? 'text-green-400' : 'text-gray-600'}`}>
        <span className={`w-2 h-2 rounded-full ${active ? 'bg-green-400' : 'bg-gray-700'}`} />
        {active ? 'Active' : 'Inactive'}
      </span>
    </div>
  )
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-500 text-xs">{label}</span>
      <span className="text-gray-300 text-xs font-mono">{value}</span>
    </div>
  )
}
