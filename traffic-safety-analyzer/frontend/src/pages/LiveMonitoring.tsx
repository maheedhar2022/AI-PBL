import { useEffect, useState, useCallback, useRef } from 'react'
import {
  Play, Square, Camera, Video, AlertTriangle, Car, Bike, Bus, Truck, Users, Activity,
  Upload, Film, Image as ImageIcon, Trash2, Download, X, Clock, Disc
} from 'lucide-react'
import {
  startMonitoring, stopMonitoring, getMonitoringStatus, getLatestFrame, configureCamera,
  uploadVideoFile, startRecording, stopRecording, listRecordings, deleteRecording,
  takeSnapshot, listSnapshots, deleteSnapshot
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'

interface VehicleCounts {
  cars?: number
  motorcycles?: number
  buses?: number
  trucks?: number
  bicycles?: number
  pedestrians?: number
}

interface PipelineMetrics {
  inference_fps?: number
  last_inference_ms?: number
  frames_processed?: number
  incidents_created?: number
  active_vehicles?: number
  vehicle_counts?: VehicleCounts
  traffic_density?: string
}

interface WsEvent {
  type: string
  metrics?: PipelineMetrics
  incident_id?: string
  violation_type?: string
  confidence?: number
}

interface RecordingItem {
  filename: string
  size_kb: number
  created_at: string
}

interface SnapshotItem {
  filename: string
  size_kb: number
  created_at: string
}

const PRESETS = [
  {
    name: '🛣️ Road Traffic CCTV (SF Street)',
    path: 'research/datasets/real_cctv_traffic.webm',
    desc: 'Real open-source street traffic CCTV with moving cars, buses, trucks, bikes',
  },
  {
    name: '🏙️ Urban District CCTV (High-View)',
    path: 'research/datasets/motorcycle_traffic_cctv.webm',
    desc: 'Real open-source high-angle urban roadway CCTV feed with dense vehicle flow',
  },
  {
    name: '🏍️ Motorcycle Safety Scene',
    path: 'research/datasets/test_video.mp4',
    desc: 'Simulated motorcycle riders for violation rules verification',
  },
  {
    name: '💻 Local Webcam',
    path: '0',
    desc: 'Live video capture from system device index 0',
  },
]

export default function LiveMonitoring() {
  const [running, setRunning] = useState(false)
  const [frame, setFrame] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<PipelineMetrics>({})
  const [alerts, setAlerts] = useState<WsEvent[]>([])
  const [cameraSource, setCameraSource] = useState('research/datasets/real_cctv_traffic.webm')
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<string | null>(null)

  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  const [recSeconds, setRecSeconds] = useState(0)
  const recTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Library / Upload modals
  const [showLibrary, setShowLibrary] = useState(false)
  const [libraryTab, setLibraryTab] = useState<'recordings' | 'snapshots'>('recordings')
  const [recordings, setRecordings] = useState<RecordingItem[]>([])
  const [snapshots, setSnapshots] = useState<SnapshotItem[]>([])
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const frameTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const showToastMsg = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 4000)
  }

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getMonitoringStatus()
      setRunning(status.pipeline_running)
      setIsRecording(status.is_recording || false)
      if (status.metrics) setMetrics(status.metrics)
    } catch { /* backend not ready */ }
  }, [])

  const fetchFrame = useCallback(async () => {
    if (!running) return
    try {
      const data = await getLatestFrame()
      if (data.frame) setFrame(data.frame)
    } catch { /* no frame yet */ }
  }, [running])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  useEffect(() => {
    if (running) {
      frameTimerRef.current = setInterval(fetchFrame, 150)
    } else {
      if (frameTimerRef.current) clearInterval(frameTimerRef.current)
    }
    return () => { if (frameTimerRef.current) clearInterval(frameTimerRef.current) }
  }, [running, fetchFrame])

  // Timer for recording duration
  useEffect(() => {
    if (isRecording) {
      recTimerRef.current = setInterval(() => {
        setRecSeconds(s => s + 1)
      }, 1000)
    } else {
      if (recTimerRef.current) clearInterval(recTimerRef.current)
      setRecSeconds(0)
    }
    return () => { if (recTimerRef.current) clearInterval(recTimerRef.current) }
  }, [isRecording])

  useWebSocket(
    useCallback((data: Record<string, unknown>) => {
      const ev = data as unknown as WsEvent
      if (ev.type === 'frame_update' && ev.metrics) {
        setMetrics(ev.metrics)
      } else if (ev.type === 'new_incident') {
        setAlerts(prev => [ev, ...prev].slice(0, 10))
      }
    }, []),
    running
  )

  const handleStart = async () => {
    setError(null)
    try {
      if (cameraSource) {
        await configureCamera(cameraSource)
      }
      await startMonitoring()
      setRunning(true)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setError(err?.response?.data?.detail ?? 'Failed to start monitoring.')
    }
  }

  const handleStop = async () => {
    if (isRecording) {
      await handleStopRecording()
    }
    await stopMonitoring()
    setRunning(false)
    setFrame(null)
  }

  const handleTakeSnapshot = () => {
    if (!frame) return
    const a = document.createElement('a')
    a.href = `data:image/jpeg;base64,${frame}`
    a.download = `cctv_snapshot_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const handleServerSnapshot = async () => {
    try {
      const res = await takeSnapshot()
      showToastMsg(`Snapshot saved to server: ${res.snapshot?.filename}`)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setError(err?.response?.data?.detail ?? 'Failed to save snapshot.')
    }
  }

  const handleStartRecording = async () => {
    try {
      const res = await startRecording()
      setIsRecording(true)
      showToastMsg(`Recording started: ${res.filename}`)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setError(err?.response?.data?.detail ?? 'Failed to start recording.')
    }
  }

  const handleStopRecording = async () => {
    try {
      const res = await stopRecording()
      setIsRecording(false)
      showToastMsg(
        `Recording saved: ${res.recording?.filename} (${res.recording?.duration_sec}s, ${res.recording?.size_kb} KB)`
      )
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setError(err?.response?.data?.detail ?? 'Failed to stop recording.')
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const res = await uploadVideoFile(file)
      setCameraSource(res.source)
      showToastMsg(`Uploaded ${res.filename}. Selected as camera source.`)
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } }
      setError(e?.response?.data?.detail ?? 'Failed to upload video.')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const loadLibraryData = async () => {
    try {
      const [recs, snaps] = await Promise.all([listRecordings(), listSnapshots()])
      setRecordings(recs.recordings || [])
      setSnapshots(snaps.snapshots || [])
    } catch { /* ignore */ }
  }

  const handleOpenLibrary = () => {
    loadLibraryData()
    setShowLibrary(true)
  }

  const handleDeleteRecording = async (filename: string) => {
    if (!confirm(`Delete recording ${filename}?`)) return
    try {
      await deleteRecording(filename)
      setRecordings(prev => prev.filter(r => r.filename !== filename))
      showToastMsg(`Deleted recording ${filename}`)
    } catch { /* ignore */ }
  }

  const handleDeleteSnapshot = async (filename: string) => {
    if (!confirm(`Delete snapshot ${filename}?`)) return
    try {
      await deleteSnapshot(filename)
      setSnapshots(prev => prev.filter(s => s.filename !== filename))
      showToastMsg(`Deleted snapshot ${filename}`)
    } catch { /* ignore */ }
  }

  const formatSecs = (s: number) => {
    const mins = Math.floor(s / 60)
    const secs = s % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const counts = metrics.vehicle_counts || {}

  return (
    <div className="p-6 space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Live CCTV Monitoring & Traffic Analytics</h1>
          <p className="text-gray-400 text-sm">Real-time AI vehicle detection, tracking, speed estimation, and road safety analysis</p>
        </div>
        <div className="flex items-center gap-2">
          {/* Hidden File Input for Video Upload */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".mp4,.webm,.avi,.mov,.mkv"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={running || uploading}
            className="btn-secondary text-xs flex items-center gap-1.5 py-1.5 px-3 disabled:opacity-40"
            title="Upload traffic video from local storage"
          >
            <Upload className="w-3.5 h-3.5 text-blue-400" />
            {uploading ? 'Uploading…' : 'Upload Video'}
          </button>

          <button
            onClick={handleOpenLibrary}
            className="btn-secondary text-xs flex items-center gap-1.5 py-1.5 px-3"
            title="Browse saved video recordings and snapshots"
          >
            <Film className="w-3.5 h-3.5 text-purple-400" /> Saved Clips & Gallery
          </button>

          <div className="ml-2 pl-2 border-l border-gray-800 flex items-center">
            {running
              ? <span className="flex items-center gap-1.5 text-green-400 text-sm font-medium"><span className="w-2.5 h-2.5 bg-green-400 rounded-full animate-pulse" />Live CCTV Stream</span>
              : <span className="flex items-center gap-1.5 text-gray-500 text-sm"><span className="w-2.5 h-2.5 bg-gray-600 rounded-full" />Feed Offline</span>
            }
          </div>
        </div>
      </div>

      {toast && (
        <div className="bg-blue-900/40 border border-blue-700 rounded-lg p-2.5 text-blue-200 text-xs flex items-center justify-between shadow">
          <span>{toast}</span>
          <button onClick={() => setToast(null)}><X className="w-3.5 h-3.5 text-blue-300" /></button>
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-300 text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)}><X className="w-4 h-4 text-red-300" /></button>
        </div>
      )}

      {/* Preset Camera Feed Selection */}
      <div className="flex flex-wrap items-center gap-2 bg-gray-900 border border-gray-800 p-3 rounded-xl text-xs">
        <span className="text-gray-400 font-medium mr-1 flex items-center gap-1">
          <Video className="w-3.5 h-3.5 text-blue-400" /> CCTV Feeds:
        </span>
        {PRESETS.map((p) => (
          <button
            key={p.path}
            onClick={() => {
              if (!running) setCameraSource(p.path)
            }}
            disabled={running}
            className={`px-3 py-1.5 rounded-lg border transition-colors ${
              cameraSource === p.path
                ? 'bg-blue-600/30 border-blue-500 text-blue-200'
                : 'bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-750'
            } disabled:opacity-50`}
            title={p.desc}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Main Camera Feed */}
        <div className="lg:col-span-2 card space-y-3">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={cameraSource}
              onChange={e => setCameraSource(e.target.value)}
              disabled={running}
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 disabled:opacity-50"
              placeholder="Source: 0 (webcam), path/to/video.webm, rtsp://..., or http://...m3u8"
            />
            {!running ? (
              <button onClick={handleStart} className="btn-primary flex items-center gap-2">
                <Play className="w-4 h-4" /> Start Feed
              </button>
            ) : (
              <div className="flex gap-2 flex-wrap">
                {/* Recording button */}
                {!isRecording ? (
                  <button
                    onClick={handleStartRecording}
                    className="bg-red-950/80 hover:bg-red-900 border border-red-700 text-red-200 rounded-lg px-3 py-2 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                    title="Record CCTV video clip to MP4"
                  >
                    <Disc className="w-4 h-4 text-red-400" /> Record
                  </button>
                ) : (
                  <button
                    onClick={handleStopRecording}
                    className="bg-red-600 hover:bg-red-700 text-white rounded-lg px-3 py-2 text-xs font-semibold flex items-center gap-1.5 animate-pulse transition-colors"
                    title="Stop and save video clip"
                  >
                    <Square className="w-3.5 h-3.5 fill-current" /> REC ({formatSecs(recSeconds)})
                  </button>
                )}

                <button
                  onClick={handleServerSnapshot}
                  disabled={!frame}
                  className="btn-secondary flex items-center gap-1 text-xs py-2 px-3 disabled:opacity-40"
                  title="Capture snapshot and save to server gallery"
                >
                  <Camera className="w-3.5 h-3.5 text-cyan-400" /> Snapshot
                </button>

                <button onClick={handleStop} className="btn-danger flex items-center gap-2 text-xs py-2 px-3">
                  <Square className="w-4 h-4" /> Stop Feed
                </button>
              </div>
            )}
          </div>

          {/* Video Frame Canvas */}
          <div className="relative bg-black rounded-lg overflow-hidden aspect-video flex items-center justify-center border border-gray-800">
            {frame ? (
              <img
                src={`data:image/jpeg;base64,${frame}`}
                alt="Live CCTV feed"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="flex flex-col items-center gap-3 text-gray-600">
                <Camera className="w-16 h-16 opacity-40" />
                <span className="text-sm font-medium">{running ? 'Processing video stream…' : 'Select a CCTV feed above and click Start'}</span>
                <span className="text-xs text-gray-700 max-w-sm text-center">Open-Source CCTV feed loops continuously with full vehicle detection, tracking, and speed analysis.</span>
              </div>
            )}
            {running && (
              <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-red-600/90 rounded-md px-2 py-0.5 text-xs text-white shadow">
                <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                CCTV ON AIR
              </div>
            )}
            {isRecording && (
              <div className="absolute top-3 left-28 flex items-center gap-1.5 bg-black/80 border border-red-500 rounded-md px-2 py-0.5 text-xs text-red-400 font-mono shadow">
                <span className="w-2 h-2 bg-red-500 rounded-full animate-ping" />
                REC {formatSecs(recSeconds)}
              </div>
            )}
            {running && metrics.traffic_density && (
              <div className="absolute top-3 right-3 bg-gray-900/80 border border-gray-700 rounded-md px-2.5 py-0.5 text-xs text-gray-200">
                Traffic: <span className={
                  metrics.traffic_density === 'Light' ? 'text-green-400 font-bold'
                  : metrics.traffic_density === 'Moderate' ? 'text-yellow-400 font-bold'
                  : 'text-red-400 font-bold'
                }>{metrics.traffic_density}</span>
              </div>
            )}
          </div>

          {/* Live Performance Bar */}
          <div className="flex gap-4 flex-wrap text-xs text-gray-500 pt-1">
            <span>Inference FPS: <b className="text-gray-300">{metrics.inference_fps?.toFixed(1) ?? '—'}</b></span>
            <span>Latency: <b className="text-gray-300">{metrics.last_inference_ms?.toFixed(0) ?? '—'} ms</b></span>
            <span>Frames: <b className="text-gray-300">{metrics.frames_processed?.toLocaleString() ?? '—'}</b></span>
            <span>Total Alerts: <b className="text-gray-300">{metrics.incidents_created ?? '—'}</b></span>
          </div>

          {/* Live Traffic Vehicle Analytics Breakdown */}
          <div className="border-t border-gray-800 pt-3">
            <div className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-1">
              <Activity className="w-3.5 h-3.5 text-blue-400" /> Active Traffic Breakdown (Current View)
            </div>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-center">
              <div className="bg-gray-800/80 border border-gray-750 p-2 rounded-lg">
                <Car className="w-4 h-4 mx-auto mb-1 text-cyan-400" />
                <div className="text-lg font-bold text-white">{counts.cars ?? 0}</div>
                <div className="text-[10px] text-gray-400">Cars</div>
              </div>
              <div className="bg-gray-800/80 border border-gray-750 p-2 rounded-lg">
                <Bike className="w-4 h-4 mx-auto mb-1 text-green-400" />
                <div className="text-lg font-bold text-white">{counts.motorcycles ?? 0}</div>
                <div className="text-[10px] text-gray-400">Motorcycles</div>
              </div>
              <div className="bg-gray-800/80 border border-gray-750 p-2 rounded-lg">
                <Bus className="w-4 h-4 mx-auto mb-1 text-amber-400" />
                <div className="text-lg font-bold text-white">{counts.buses ?? 0}</div>
                <div className="text-[10px] text-gray-400">Buses</div>
              </div>
              <div className="bg-gray-800/80 border border-gray-750 p-2 rounded-lg">
                <Truck className="w-4 h-4 mx-auto mb-1 text-purple-400" />
                <div className="text-lg font-bold text-white">{counts.trucks ?? 0}</div>
                <div className="text-[10px] text-gray-400">Trucks</div>
              </div>
              <div className="bg-gray-800/80 border border-gray-750 p-2 rounded-lg">
                <Bike className="w-4 h-4 mx-auto mb-1 text-lime-400" />
                <div className="text-lg font-bold text-white">{counts.bicycles ?? 0}</div>
                <div className="text-[10px] text-gray-400">Bicycles</div>
              </div>
              <div className="bg-gray-800/80 border border-gray-750 p-2 rounded-lg">
                <Users className="w-4 h-4 mx-auto mb-1 text-orange-400" />
                <div className="text-lg font-bold text-white">{counts.pedestrians ?? 0}</div>
                <div className="text-[10px] text-gray-400">Pedestrians</div>
              </div>
            </div>
          </div>
        </div>

        {/* Live Alerts & Incident Log Panel */}
        <div className="card space-y-3 flex flex-col">
          <h2 className="font-semibold text-white flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-orange-400" />
            Safety Alerts & Violations
          </h2>
          {alerts.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-gray-600 text-sm p-6 text-center">
              <AlertTriangle className="w-8 h-8 opacity-20 mb-2" />
              <span>Monitoring active road feed.</span>
              <span className="text-xs text-gray-700 mt-1">Detections meeting temporal confidence criteria will populate here.</span>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto flex-1">
              {alerts.map((ev, i) => (
                <div key={i} className="bg-gray-800 border border-gray-700/60 rounded-lg p-3 space-y-1 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className={ev.violation_type === 'helmet_violation' ? 'badge-helmet' : 'badge badge-triple'}>
                      {ev.violation_type === 'helmet_violation' ? 'Helmet Alert' : 'Triple Riding'}
                    </span>
                    <span className="text-xs text-gray-400 font-mono">{((ev.confidence ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="text-xs text-gray-500 font-mono truncate">Incident: {ev.incident_id?.slice(0, 8)}</div>
                </div>
              ))}
            </div>
          )}
          <div className="text-xs text-gray-600 pt-2 border-t border-gray-800">
            Recorded incidents saved to database with evidence crops.
          </div>
        </div>
      </div>

      {/* Saved Clips & Snapshots Gallery Modal */}
      {showLibrary && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm">
          <div className="bg-gray-900 border border-gray-800 rounded-2xl w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-800">
              <div className="flex items-center gap-2">
                <Film className="w-5 h-5 text-purple-400" />
                <h3 className="font-bold text-white text-base">Evidence Library & Captured Media</h3>
              </div>
              <button
                onClick={() => setShowLibrary(false)}
                className="text-gray-400 hover:text-white p-1 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Tabs */}
            <div className="flex border-b border-gray-800 bg-gray-950/50 px-4">
              <button
                onClick={() => setLibraryTab('recordings')}
                className={`py-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 transition-colors ${
                  libraryTab === 'recordings'
                    ? 'border-purple-500 text-purple-300'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                <Film className="w-4 h-4" /> Video Clips ({recordings.length})
              </button>
              <button
                onClick={() => setLibraryTab('snapshots')}
                className={`py-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 transition-colors ${
                  libraryTab === 'snapshots'
                    ? 'border-cyan-500 text-cyan-300'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                <ImageIcon className="w-4 h-4" /> Snapshots ({snapshots.length})
              </button>
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {libraryTab === 'recordings' ? (
                recordings.length === 0 ? (
                  <div className="text-center py-12 text-gray-600 text-sm">
                    <Film className="w-10 h-10 mx-auto opacity-30 mb-2" />
                    No video clips recorded yet. Click "Record" during live monitoring.
                  </div>
                ) : (
                  <div className="divide-y divide-gray-800">
                    {recordings.map((rec) => (
                      <div key={rec.filename} className="py-2.5 flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2.5">
                          <Film className="w-4 h-4 text-purple-400 shrink-0" />
                          <div>
                            <div className="text-gray-200 font-medium">{rec.filename}</div>
                            <div className="text-gray-500 text-[11px]">
                              {rec.size_kb} KB • {new Date(rec.created_at).toLocaleString()}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <a
                            href={`/api/v1/monitoring/recordings/${rec.filename}`}
                            download
                            className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1"
                          >
                            <Download className="w-3 h-3 text-green-400" /> Download
                          </a>
                          <button
                            onClick={() => handleDeleteRecording(rec.filename)}
                            className="text-gray-600 hover:text-red-400 p-1"
                            title="Delete clip"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )
              ) : snapshots.length === 0 ? (
                <div className="text-center py-12 text-gray-600 text-sm">
                  <ImageIcon className="w-10 h-10 mx-auto opacity-30 mb-2" />
                  No snapshots captured yet. Click "Snapshot" during live monitoring.
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {snapshots.map((snap) => (
                    <div key={snap.filename} className="bg-gray-800/60 border border-gray-700/60 rounded-xl overflow-hidden shadow-sm flex flex-col">
                      <img
                        src={`/evidence/snapshots/${snap.filename}`}
                        alt={snap.filename}
                        className="w-full aspect-video object-cover bg-black"
                      />
                      <div className="p-2 text-[11px] flex-1 flex flex-col justify-between space-y-1">
                        <div className="text-gray-300 font-mono truncate">{snap.filename}</div>
                        <div className="text-gray-500">{snap.size_kb} KB</div>
                        <div className="flex items-center justify-between pt-1 border-t border-gray-700/40">
                          <a
                            href={`/evidence/snapshots/${snap.filename}`}
                            download
                            className="text-green-400 hover:underline flex items-center gap-1"
                          >
                            <Download className="w-3 h-3" /> Save
                          </a>
                          <button
                            onClick={() => handleDeleteSnapshot(snap.filename)}
                            className="text-gray-500 hover:text-red-400"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
