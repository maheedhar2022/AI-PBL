import { useEffect, useState, useCallback } from 'react'
import { Search, Filter, ChevronLeft, ChevronRight, ExternalLink, Trash2 } from 'lucide-react'
import { listIncidents, reviewIncident, deleteIncident } from '../services/api'
import { format } from 'date-fns'

interface Incident {
  id: string
  violation_type: string
  event_timestamp: string
  camera_source: string
  track_id: number | null
  confidence: number
  detection_metadata: Record<string, unknown>
  evidence_image_path: string | null
  review_status: string
  reviewer_notes: string | null
  created_at: string
  outcome: string
  rider_count: number | null
}

const STATUSES = ['', 'pending', 'confirmed', 'dismissed']
const TYPES = ['', 'helmet_violation', 'triple_riding']

export default function IncidentExplorer() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [filterType, setFilterType] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [selected, setSelected] = useState<Incident | null>(null)
  const [loading, setLoading] = useState(false)
  const PAGE_SIZE = 15

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await listIncidents({
        violation_type: filterType || undefined,
        review_status: filterStatus || undefined,
        page,
        page_size: PAGE_SIZE,
      })
      setIncidents(data.items)
      setTotal(data.total)
    } catch { /* ignore */ }
    setLoading(false)
  }, [page, filterType, filterStatus])

  useEffect(() => { load() }, [load])

  const handleReview = async (id: string, status: string) => {
    await reviewIncident(id, status)
    load()
    if (selected?.id === id) setSelected(prev => prev ? { ...prev, review_status: status } : null)
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this incident?')) return
    await deleteIncident(id)
    if (selected?.id === id) setSelected(null)
    load()
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="p-6 h-full flex flex-col gap-4">
      <div>
        <h1 className="text-2xl font-bold text-white">Incident Explorer</h1>
        <p className="text-gray-400 text-sm">{total} incidents recorded</p>
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <select
          value={filterType}
          onChange={e => { setFilterType(e.target.value); setPage(1) }}
          className="bg-gray-800 border border-gray-700 text-gray-300 rounded-lg px-3 py-2 text-sm"
        >
          <option value="">All Types</option>
          <option value="helmet_violation">Helmet Violation</option>
          <option value="triple_riding">Triple Riding</option>
        </select>
        <select
          value={filterStatus}
          onChange={e => { setFilterStatus(e.target.value); setPage(1) }}
          className="bg-gray-800 border border-gray-700 text-gray-300 rounded-lg px-3 py-2 text-sm"
        >
          <option value="">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="confirmed">Confirmed</option>
          <option value="dismissed">Dismissed</option>
        </select>
      </div>

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Incident List */}
        <div className="flex-1 card overflow-y-auto">
          {loading && <div className="text-gray-500 text-sm">Loading…</div>}
          {!loading && incidents.length === 0 && (
            <div className="text-gray-500 text-sm">No incidents match the filters.</div>
          )}
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs border-b border-gray-800">
                <th className="text-left py-2">Type</th>
                <th className="text-left py-2">Timestamp</th>
                <th className="text-left py-2">Confidence</th>
                <th className="text-left py-2">Status</th>
                <th className="text-left py-2"></th>
              </tr>
            </thead>
            <tbody>
              {incidents.map(inc => (
                <tr
                  key={inc.id}
                  onClick={() => setSelected(inc)}
                  className={`border-b border-gray-800 cursor-pointer hover:bg-gray-800 transition-colors ${selected?.id === inc.id ? 'bg-gray-800/80' : ''}`}
                >
                  <td className="py-2.5">
                    {inc.violation_type === 'helmet_violation'
                      ? <span className="badge-helmet">Helmet</span>
                      : <span className="badge badge-triple">Triple</span>}
                  </td>
                  <td className="py-2.5 text-gray-400 text-xs">
                    {format(new Date(inc.event_timestamp), 'MMM d, HH:mm:ss')}
                  </td>
                  <td className="py-2.5 text-gray-300 font-mono text-xs">
                    {(inc.confidence * 100).toFixed(0)}%
                  </td>
                  <td className="py-2.5">
                    <span className={`badge text-xs ${
                      inc.review_status === 'pending' ? 'badge-pending' :
                      inc.review_status === 'confirmed' ? 'badge-confirmed' :
                      'bg-gray-800 text-gray-400'
                    }`}>{inc.review_status}</span>
                  </td>
                  <td className="py-2.5">
                    <button onClick={(e) => { e.stopPropagation(); handleDelete(inc.id) }}
                      className="text-gray-600 hover:text-red-400 transition-colors">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          <div className="flex items-center justify-between pt-3 border-t border-gray-800 mt-2">
            <span className="text-xs text-gray-600">Page {page} of {totalPages}</span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="btn-secondary py-1 px-2 text-xs disabled:opacity-50"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="btn-secondary py-1 px-2 text-xs disabled:opacity-50"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Detail Panel */}
        <div className="w-80 card overflow-y-auto shrink-0">
          {!selected ? (
            <div className="text-gray-600 text-sm flex flex-col items-center justify-center h-full gap-2">
              <Search className="w-8 h-8" />
              <span>Select an incident to view details</span>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-white">Incident Detail</h3>
                <div className="text-xs text-gray-600 font-mono mt-0.5">{selected.id}</div>
              </div>

              <div className="space-y-2 text-sm">
                <Row label="Type" value={selected.violation_type.replace('_', ' ')} />
                <Row label="Outcome" value={selected.outcome.replace('_', ' ')} />
                <Row label="Timestamp" value={format(new Date(selected.event_timestamp), 'PPpp')} />
                <Row label="Camera" value={selected.camera_source} />
                <Row label="Track ID" value={selected.track_id?.toString() ?? 'N/A'} />
                <Row label="Confidence" value={`${(selected.confidence * 100).toFixed(1)}%`} />
                {selected.rider_count && <Row label="Rider Count" value={selected.rider_count.toString()} />}
              </div>

              {/* Evidence Image */}
              {selected.evidence_image_path && (
                <div>
                  <div className="text-xs text-gray-500 mb-1">Evidence Image</div>
                  <img
                    src={`/evidence/${selected.evidence_image_path.split('/').pop()}`}
                    alt="Evidence"
                    className="w-full rounded-lg border border-gray-700"
                    onError={(e) => (e.currentTarget.style.display = 'none')}
                  />
                </div>
              )}

              {/* Review actions */}
              <div className="space-y-2 pt-2 border-t border-gray-800">
                <div className="text-xs text-gray-500">Review Status: <b>{selected.review_status}</b></div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleReview(selected.id, 'confirmed')}
                    disabled={selected.review_status === 'confirmed'}
                    className="btn-danger text-xs py-1 flex-1 disabled:opacity-40"
                  >Confirm</button>
                  <button
                    onClick={() => handleReview(selected.id, 'dismissed')}
                    disabled={selected.review_status === 'dismissed'}
                    className="btn-secondary text-xs py-1 flex-1 disabled:opacity-40"
                  >Dismiss</button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-gray-500 shrink-0">{label}</span>
      <span className="text-gray-200 text-right truncate">{value}</span>
    </div>
  )
}

