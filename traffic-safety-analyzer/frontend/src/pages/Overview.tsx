import { useEffect, useState, useCallback } from 'react'
import {
  AlertTriangle, Video, Clock, CheckSquare,
  Activity, RefreshCw
} from 'lucide-react'
import { getDashboardStats, getRecentIncidents } from '../services/api'
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

export default function Overview() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [recent, setRecent] = useState<RecentIncident[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const [s, r] = await Promise.all([getDashboardStats(), getRecentIncidents(8)])
      setStats(s)
      setRecent(r)
      setError(null)
    } catch (e: unknown) {
      setError('Cannot reach API. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
    const timer = setInterval(load, 10000)
    return () => clearInterval(timer)
  }, [load])

  const violationBadge = (type: string) =>
    type === 'helmet_violation'
      ? <span className="badge-helmet">Helmet</span>
      : <span className="badge badge-triple">Triple Riding</span>

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Overview</h1>
          <p className="text-gray-400 text-sm mt-0.5">System summary and recent activity</p>
        </div>
        <button onClick={load} className="btn-secondary flex items-center gap-2 text-sm">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<AlertTriangle className="w-5 h-5 text-orange-400" />}
          label="Total Incidents"
          value={stats?.total_incidents ?? '—'}
          sub="all time"
        />
        <StatCard
          icon={<Activity className="w-5 h-5 text-red-400" />}
          label="Helmet Violations"
          value={stats?.helmet_incidents ?? '—'}
          sub="potential"
        />
        <StatCard
          icon={<Video className="w-5 h-5 text-purple-400" />}
          label="Triple Riding"
          value={stats?.triple_riding_incidents ?? '—'}
          sub="potential"
        />
        <StatCard
          icon={<CheckSquare className="w-5 h-5 text-yellow-400" />}
          label="Pending Review"
          value={stats?.pending_review ?? '—'}
          sub="incidents"
        />
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card">
          <h2 className="font-semibold text-white mb-3 flex items-center gap-2">
            <Clock className="w-4 h-4 text-blue-400" />
            System Status
          </h2>
          <div className="space-y-3">
            <StatusRow label="Camera" active={stats?.camera_running ?? false} />
            <StatusRow label="AI Pipeline" active={stats?.pipeline_running ?? false} />
            <div className="border-t border-gray-800 pt-3 space-y-1.5">
              <MetricRow label="Frames Processed" value={stats?.pipeline_metrics?.frames_processed?.toLocaleString() ?? '0'} />
              <MetricRow label="Inference FPS" value={`${stats?.pipeline_metrics?.inference_fps?.toFixed(1) ?? '0'} fps`} />
              <MetricRow label="Last Latency" value={`${stats?.pipeline_metrics?.last_inference_ms?.toFixed(0) ?? '0'} ms`} />
              <MetricRow label="Incidents Created" value={`${stats?.pipeline_metrics?.incidents_created ?? 0}`} />
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card">
          <h2 className="font-semibold text-white mb-3">Recent Activity</h2>
          {loading ? (
            <div className="text-gray-500 text-sm">Loading…</div>
          ) : recent.length === 0 ? (
            <div className="text-gray-500 text-sm">No incidents recorded yet.</div>
          ) : (
            <div className="space-y-2">
              {recent.map(inc => (
                <div key={inc.id} className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
                  <div className="flex items-center gap-2">
                    {violationBadge(inc.violation_type)}
                    <span className="text-gray-400 text-xs">
                      {formatDistanceToNow(new Date(inc.event_timestamp), { addSuffix: true })}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500">{(inc.confidence * 100).toFixed(0)}%</span>
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

function StatCard({ icon, label, value, sub }: {
  icon: React.ReactNode
  label: string
  value: number | string
  sub: string
}) {
  return (
    <div className="stat-card">
      <div className="flex items-center gap-2 text-gray-400 text-sm">
        {icon}
        <span>{label}</span>
      </div>
      <div className="text-3xl font-bold text-white">{value}</div>
      <div className="text-gray-600 text-xs">{sub}</div>
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

