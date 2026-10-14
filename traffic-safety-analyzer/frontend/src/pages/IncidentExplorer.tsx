import { useEffect, useState, useCallback } from 'react'
import {
  Search, Filter, ChevronLeft, ChevronRight,
  Trash2, Download, FileSpreadsheet, FileText, Check, X, CheckSquare, Square
} from 'lucide-react'
import {
  listIncidents, reviewIncident, deleteIncident, exportIncidents,
  batchReviewIncidents, bulkDeleteIncidents
} from '../services/api'
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

export default function IncidentExplorer() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [filterType, setFilterType] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [selected, setSelected] = useState<Incident | null>(null)
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [reviewerNotes, setReviewerNotes] = useState('')
  const [loading, setLoading] = useState(false)
  const [batchLoading, setBatchLoading] = useState(false)
  const [exporting, setExporting] = useState(false)
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

  useEffect(() => {
    if (selected) {
      setReviewerNotes(selected.reviewer_notes || '')
    }
  }, [selected])

  const handleReview = async (id: string, status: string) => {
    await reviewIncident(id, status, reviewerNotes || undefined)
    load()
    if (selected?.id === id) {
      setSelected(prev => prev ? { ...prev, review_status: status, reviewer_notes: reviewerNotes } : null)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this incident record?')) return
    await deleteIncident(id)
    if (selected?.id === id) setSelected(null)
    load()
  }

  const handleExport = async (formatType: 'csv' | 'json') => {
    setExporting(true)
    try {
      await exportIncidents({
        violation_type: filterType || undefined,
        review_status: filterStatus || undefined,
        format: formatType,
      })
    } catch { /* ignore */ }
    setExporting(false)
  }

  const handleToggleSelectAll = () => {
    if (selectedIds.length === filteredIncidents.length && filteredIncidents.length > 0) {
      setSelectedIds([])
    } else {
      setSelectedIds(filteredIncidents.map(i => i.id))
    }
  }

  const handleToggleSelect = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  const handleBatchReview = async (status: string) => {
    if (selectedIds.length === 0) return
    setBatchLoading(true)
    try {
      await batchReviewIncidents(selectedIds, status, reviewerNotes || undefined)
      setSelectedIds([])
      await load()
    } catch { /* ignore */ }
    setBatchLoading(false)
  }

  const handleBulkDelete = async () => {
    if (selectedIds.length === 0) return
    if (!confirm(`Permanently delete ${selectedIds.length} selected incidents?`)) return
    setBatchLoading(true)
    try {
      await bulkDeleteIncidents(selectedIds)
      setSelectedIds([])
      if (selected && selectedIds.includes(selected.id)) setSelected(null)
      await load()
    } catch { /* ignore */ }
    setBatchLoading(false)
  }

  const filteredIncidents = incidents.filter(inc => {
    if (!searchQuery) return true
    const q = searchQuery.toLowerCase()
    return (
      inc.id.toLowerCase().includes(q) ||
      (inc.track_id !== null && inc.track_id.toString().includes(q)) ||
      inc.camera_source.toLowerCase().includes(q)
    )
  })

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="p-6 h-full flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Incident Explorer & Review Audit</h1>
          <p className="text-gray-400 text-sm">{total} potential violations recorded across all sources</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleExport('csv')}
            disabled={exporting || total === 0}
            className="btn-secondary text-xs flex items-center gap-1.5 py-1.5 px-3 disabled:opacity-40"
            title="Download CSV report"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-green-400" /> Export CSV
          </button>
          <button
            onClick={() => handleExport('json')}
            disabled={exporting || total === 0}
            className="btn-secondary text-xs flex items-center gap-1.5 py-1.5 px-3 disabled:opacity-40"
            title="Download JSON report"
          >
            <FileText className="w-3.5 h-3.5 text-blue-400" /> Export JSON
          </button>
        </div>
      </div>

      {/* Filters & Search */}
      <div className="flex gap-3 flex-wrap items-center">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="w-4 h-4 text-gray-500 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by Track ID, Incident ID, or Source..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 text-gray-200 rounded-lg pl-9 pr-3 py-2 text-sm placeholder-gray-500"
          />
        </div>
        <select
          value={filterType}
          onChange={e => { setFilterType(e.target.value); setPage(1) }}
          className="bg-gray-800 border border-gray-700 text-gray-300 rounded-lg px-3 py-2 text-sm"
        >
          <option value="">All Violation Types</option>
          <option value="helmet_violation">Helmet Violation</option>
          <option value="triple_riding">Triple Riding</option>
        </select>
        <select
          value={filterStatus}
          onChange={e => { setFilterStatus(e.target.value); setPage(1) }}
          className="bg-gray-800 border border-gray-700 text-gray-300 rounded-lg px-3 py-2 text-sm"
        >
          <option value="">All Review Statuses</option>
          <option value="pending">Pending</option>
          <option value="confirmed">Confirmed</option>
          <option value="dismissed">Dismissed</option>
        </select>
      </div>

      {/* Bulk Action Toolbar */}
      {selectedIds.length > 0 && (
        <div className="bg-blue-950/40 border border-blue-800/80 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 shadow-md">
          <div className="flex items-center gap-2 text-sm text-blue-200">
            <CheckSquare className="w-4 h-4 text-blue-400" />
            <span className="font-semibold">{selectedIds.length}</span> incident{selectedIds.length > 1 ? 's' : ''} selected
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleBatchReview('confirmed')}
              disabled={batchLoading}
              className="btn-danger text-xs py-1.5 px-3 flex items-center gap-1.5 disabled:opacity-50"
            >
              <Check className="w-3.5 h-3.5" /> Bulk Confirm
            </button>
            <button
              onClick={() => handleBatchReview('dismissed')}
              disabled={batchLoading}
              className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5 disabled:opacity-50"
            >
              <X className="w-3.5 h-3.5" /> Bulk Dismiss
            </button>
            <button
              onClick={handleBulkDelete}
              disabled={batchLoading}
              className="bg-red-900/60 hover:bg-red-800 text-red-200 border border-red-700 rounded-lg text-xs py-1.5 px-3 flex items-center gap-1.5 transition-colors disabled:opacity-50"
            >
              <Trash2 className="w-3.5 h-3.5" /> Bulk Delete
            </button>
            <button
              onClick={() => setSelectedIds([])}
              className="text-xs text-gray-400 hover:text-gray-200 underline ml-2"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Incident List */}
        <div className="flex-1 card overflow-y-auto">
          {loading && <div className="text-gray-500 text-sm">Loading incidents…</div>}
          {!loading && filteredIncidents.length === 0 && (
            <div className="text-gray-500 text-sm py-8 text-center">
              No incidents match the selected filters or search query.
            </div>
          )}
          {filteredIncidents.length > 0 && (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-gray-800">
                  <th className="w-8 py-2 text-center">
                    <input
                      type="checkbox"
                      checked={selectedIds.length === filteredIncidents.length && filteredIncidents.length > 0}
                      onChange={handleToggleSelectAll}
                      className="rounded border-gray-700 bg-gray-800 text-blue-500 focus:ring-0 cursor-pointer"
                      title="Select all visible incidents"
                    />
                  </th>
                  <th className="text-left py-2 font-medium">Type</th>
                  <th className="text-left py-2 font-medium">Track ID</th>
                  <th className="text-left py-2 font-medium">Timestamp</th>
                  <th className="text-left py-2 font-medium">Confidence</th>
                  <th className="text-left py-2 font-medium">Status</th>
                  <th className="text-left py-2 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {filteredIncidents.map(inc => {
                  const isChecked = selectedIds.includes(inc.id)
                  return (
                    <tr
                      key={inc.id}
                      onClick={() => setSelected(inc)}
                      className={`border-b border-gray-800/80 cursor-pointer hover:bg-gray-800/70 transition-colors ${
                        selected?.id === inc.id ? 'bg-gray-800' : isChecked ? 'bg-blue-950/20' : ''
                      }`}
                    >
                      <td className="py-2.5 w-8 text-center" onClick={(e) => handleToggleSelect(inc.id, e)}>
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => {}}
                          className="rounded border-gray-700 bg-gray-800 text-blue-500 focus:ring-0 cursor-pointer"
                        />
                      </td>
                      <td className="py-2.5">
                        {inc.violation_type === 'helmet_violation'
                          ? <span className="badge-helmet">Helmet</span>
                          : <span className="badge badge-triple">Triple</span>}
                      </td>
                      <td className="py-2.5 font-mono text-xs text-yellow-400">
                        {inc.track_id !== null ? `#${inc.track_id}` : '—'}
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
                      <td className="py-2.5 text-right">
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDelete(inc.id) }}
                          className="text-gray-600 hover:text-red-400 transition-colors p-1"
                          title="Delete incident"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}

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

        {/* Detail & Review Panel */}
        <div className="w-84 card overflow-y-auto shrink-0 space-y-4">
          {!selected ? (
            <div className="text-gray-600 text-sm flex flex-col items-center justify-center h-full gap-2">
              <Search className="w-8 h-8 opacity-30" />
              <span>Select an incident to view details</span>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-white">Incident Details</h3>
                <div className="text-xs text-gray-500 font-mono mt-0.5 break-all">{selected.id}</div>
              </div>

              <div className="space-y-2 text-sm border-t border-gray-800 pt-2">
                <Row label="Violation" value={selected.violation_type.replace('_', ' ')} />
                <Row label="Outcome" value={selected.outcome.replace('_', ' ')} />
                <Row label="Timestamp" value={format(new Date(selected.event_timestamp), 'PPpp')} />
                <Row label="Camera Source" value={selected.camera_source} />
                <Row label="Track ID" value={selected.track_id?.toString() ?? 'N/A'} />
                <Row label="AI Confidence" value={`${(selected.confidence * 100).toFixed(1)}%`} />
                {selected.rider_count !== null && (
                  <Row label="Rider Count" value={selected.rider_count.toString()} />
                )}
              </div>

              {/* Evidence Image */}
              {selected.evidence_image_path && (
                <div className="space-y-1">
                  <div className="text-xs text-gray-500 font-medium">Evidence Frame Crop</div>
                  <img
                    src={`/evidence/${selected.evidence_image_path.replace(/\\/g, '/').split('/').pop()}`}
                    alt="Evidence"
                    className="w-full rounded-lg border border-gray-700 max-h-48 object-cover bg-black"
                    onError={(e) => (e.currentTarget.style.display = 'none')}
                  />
                </div>
              )}

              {/* Reviewer Notes & Action Buttons */}
              <div className="space-y-2 pt-2 border-t border-gray-800">
                <div className="text-xs text-gray-400 font-medium">Review Audit Trail</div>
                <div className="text-xs text-gray-500">Status: <b className="text-gray-300 uppercase">{selected.review_status}</b></div>
                <textarea
                  placeholder="Optional review audit notes..."
                  value={reviewerNotes}
                  onChange={e => setReviewerNotes(e.target.value)}
                  rows={2}
                  className="w-full bg-gray-800 border border-gray-700 text-xs text-gray-200 rounded-lg p-2 placeholder-gray-500"
                />
                <div className="flex gap-2">
                  <button
                    onClick={() => handleReview(selected.id, 'confirmed')}
                    className="btn-danger text-xs py-1.5 flex-1 flex items-center justify-center gap-1"
                  >
                    <Check className="w-3.5 h-3.5" /> Confirm Violation
                  </button>
                  <button
                    onClick={() => handleReview(selected.id, 'dismissed')}
                    className="btn-secondary text-xs py-1.5 flex-1 flex items-center justify-center gap-1"
                  >
                    <X className="w-3.5 h-3.5" /> Dismiss
                  </button>
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
    <div className="flex justify-between gap-2 text-xs">
      <span className="text-gray-500 shrink-0">{label}</span>
      <span className="text-gray-200 text-right truncate font-medium">{value}</span>
    </div>
  )
}
