import { useEffect, useState, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import {
  Play, Square, Camera, Video, AlertTriangle, Car, Bike, Bus, Truck, Users, Activity,
  Upload, Film, Image as ImageIcon, Trash2, Download, X, Clock, Disc,
  Maximize2, Minimize2, Volume2, VolumeX, Cpu, Layers, Zap, ShieldCheck, Eye
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
  vehicle_number?: string
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
    name: '🇮🇳 Indian Road Traffic CCTV (Link Road, Cuttack)',
    path: 'research/datasets/indian_road_traffic_cctv.webm',
    desc: 'Real open-source Indian roadway CCTV footage (Link Road, Cuttack) with motorcycles, auto-rickshaws, and cars',
  },
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
  const [cameraSource, setCameraSource] = useState('research/datasets/indian_road_traffic_cctv.webm')
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
  const [selectedSnapshot, setSelectedSnapshot] = useState<SnapshotItem | null>(null)
  const [previewRecording, setPreviewRecording] = useState<string | null>(null)
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

  // Fullscreen and Audio alerts
  const [soundEnabled, setSoundEnabled] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const videoContainerRef = useRef<HTMLDivElement | null>(null)

  const playViolationChime = useCallback(() => {
    if (!soundEnabled) return
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new AudioCtx()
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = 'sine'
      osc.frequency.setValueAtTime(880, ctx.currentTime)
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.22)
      gain.gain.setValueAtTime(0.18, ctx.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.22)
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.start()
      osc.stop(ctx.currentTime + 0.23)
    } catch { /* audio not allowed yet or unsupported */ }
  }, [soundEnabled])

  const toggleFullscreen = () => {
    if (!videoContainerRef.current) return
    if (!document.fullscreenElement) {
      videoContainerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {})
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {})
    }
  }

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener('fullscreenchange', handleFsChange)
    return () => document.removeEventListener('fullscreenchange', handleFsChange)
  }, [])

  useWebSocket(
    useCallback((data: Record<string, unknown>) => {
      const ev = data as unknown as WsEvent
      if (ev.type === 'frame_update' && ev.metrics) {
        setMetrics(ev.metrics)
      } else if (ev.type === 'new_incident') {
        setAlerts(prev => [ev, ...prev].slice(0, 10))
        playViolationChime()
      }
    }, [playViolationChime]),
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
      showToastMsg('AI Traffic Safety Monitoring started.')
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
      const filename = res.snapshot?.filename || 'snapshot.jpg'
      const resText = res.snapshot?.resolution ? ` (${res.snapshot.resolution})` : ''
      showToastMsg(`Snapshot saved: ${filename}${resText}`)
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
      if (running) {
        showToastMsg(`Uploaded ${res.filename} — live AI analysis hot-swapped to new video!`)
      } else {
        showToastMsg(`Uploaded ${res.filename}. Selected as source — click 'Start Feed'.`)
      }
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
            onClick={async () => {
              setCameraSource(p.path)
              if (running) {
                try {
                  await configureCamera(p.path)
                  showToastMsg(`Switched live feed to ${p.name}`)
                } catch (e: unknown) {
                  const err = e as { response?: { data?: { detail?: string } } }
                  setError(err?.response?.data?.detail ?? 'Failed to switch feed.')
                }
              }
            }}
            className={`px-3 py-1.5 rounded-lg border transition-colors flex items-center gap-1.5 ${
              cameraSource === p.path
                ? 'bg-blue-600/30 border-blue-500 text-blue-200 font-semibold'
                : 'bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-750'
            }`}
            title={p.desc}
          >
            <span>{p.name}</span>
            {cameraSource === p.path && running && (
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 inline-block animate-pulse" />
            )}
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
          <div
            ref={videoContainerRef}
            className="relative bg-black rounded-lg overflow-hidden aspect-video flex items-center justify-center border border-gray-800"
          >
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
            {/* Top Right Floating Toolbar (Traffic, Audio Alert Toggle, Fullscreen) */}
            <div className="absolute top-3 right-3 flex items-center gap-2">
              {running && metrics.traffic_density && (
                <div className="bg-gray-900/80 border border-gray-700 rounded-md px-2.5 py-0.5 text-xs text-gray-200">
                  Traffic: <span className={
                    metrics.traffic_density === 'Light' ? 'text-green-400 font-bold'
                    : metrics.traffic_density === 'Moderate' ? 'text-yellow-400 font-bold'
                    : 'text-red-400 font-bold'
                  }>{metrics.traffic_density}</span>
                </div>
              )}
              <button
                onClick={() => {
                  const next = !soundEnabled
                  setSoundEnabled(next)
                  showToastMsg(next ? 'Violation audio alerts enabled.' : 'Violation audio alerts muted.')
                }}
                className="bg-gray-900/80 hover:bg-gray-850 border border-gray-700 rounded-md p-1.5 text-xs transition-colors"
                title={soundEnabled ? 'Mute violation audio alerts' : 'Enable violation audio alerts'}
              >
                {soundEnabled ? <Volume2 className="w-3.5 h-3.5 text-green-400" /> : <VolumeX className="w-3.5 h-3.5 text-gray-400" />}
              </button>
              <button
                onClick={toggleFullscreen}
                className="bg-gray-900/80 hover:bg-gray-850 border border-gray-700 rounded-md p-1.5 text-xs text-gray-300 hover:text-white transition-colors"
                title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Video Feed'}
              >
                {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          {/* Active AI Neural Pipeline Badge */}
          <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-gray-900/90 border border-purple-900/50 text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
              <span className="text-gray-300 font-semibold flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-purple-400" /> Neural Vision Pipeline:
              </span>
              <span className="text-purple-300 font-mono text-[11px] hidden md:inline">
                YOLOv8-CSP (3.15M) + DeepHelmetCNN-SE + DeepSORT (128-d ReID) + ViolationMLP
              </span>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 font-mono border border-purple-800">
              3.83M DL Params
            </span>
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

        {/* Active Deep Learning Neural Pipeline Status Card */}
        <div className="card space-y-2.5 border border-purple-900/50 bg-gray-900/90 shadow-md">
          <div className="flex items-center justify-between border-b border-gray-800 pb-2">
            <h2 className="font-semibold text-white flex items-center gap-2 text-xs">
              <Cpu className="w-4 h-4 text-purple-400" />
              Active Deep Learning Models
            </h2>
            <Link to="/settings" className="text-[11px] text-purple-400 hover:text-purple-300 flex items-center gap-1 font-medium">
              Benchmark <Zap className="w-3 h-3 text-yellow-400" />
            </Link>
          </div>
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center justify-between p-2 rounded bg-gray-800/50 border border-gray-750">
              <div className="flex items-center gap-2">
                <Layers className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                <div>
                  <div className="font-semibold text-white">YOLOv8 Deep Detector</div>
                  <div className="text-[10px] text-gray-400">3.15M params · Vehicles & Pedestrians</div>
                </div>
              </div>
              <span className="badge bg-blue-950 text-blue-300 text-[10px] font-mono border border-blue-800">Active</span>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-gray-800/50 border border-gray-750">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                <div>
                  <div className="font-semibold text-white">DeepHelmetCNN-SE</div>
                  <div className="text-[10px] text-gray-400">184K params · SE Attention Head Net</div>
                </div>
              </div>
              <span className="badge bg-amber-950 text-amber-300 text-[10px] font-mono border border-amber-800">Active</span>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-gray-800/50 border border-gray-750">
              <div className="flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                <div>
                  <div className="font-semibold text-white">DeepSORT Visual Re-ID</div>
                  <div className="text-[10px] text-gray-400">480K params · 128-d Cosine Embeddings</div>
                </div>
              </div>
              <span className="badge bg-cyan-950 text-cyan-300 text-[10px] font-mono border border-cyan-800">Active</span>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-gray-800/50 border border-gray-750">
              <div className="flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-green-400 shrink-0" />
                <div>
                  <div className="font-semibold text-white">ViolationRiskMLP</div>
                  <div className="text-[10px] text-gray-400">7.7K params · Spatio-Temporal Kinematics</div>
                </div>
              </div>
              <span className="badge bg-green-950 text-green-300 text-[10px] font-mono border border-green-800">Active</span>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-gray-800/50 border border-gray-750">
              <div className="flex items-center gap-2">
                <span className="text-sm shrink-0">🇮🇳</span>
                <div>
                  <div className="font-semibold text-white">IndianANPR-HSRP OCR Net</div>
                  <div className="text-[10px] text-gray-400">71K params · MoRTH Vehicle Plate Recognition</div>
                </div>
              </div>
              <span className="badge bg-indigo-950 text-indigo-300 text-[10px] font-mono border border-indigo-800">Active</span>
            </div>
          </div>
        </div>

        {/* Live Alerts & Incident Log Panel */}
        <div className="card space-y-3 flex flex-col">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-white flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-orange-400" />
              Safety Alerts & Violations
            </h2>
            {alerts.length > 0 && (
              <button
                onClick={() => setAlerts([])}
                className="text-[11px] text-gray-500 hover:text-gray-300 underline"
                title="Clear alerts list"
              >
                Clear
              </button>
            )}
          </div>
          {alerts.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-gray-600 text-sm p-6 text-center">
              <AlertTriangle className="w-8 h-8 opacity-20 mb-2" />
              <span>Monitoring active road feed.</span>
              <span className="text-xs text-gray-700 mt-1">Detections meeting temporal confidence criteria will populate here.</span>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto flex-1">
              {alerts.map((ev, i) => (
                <div key={i} className="bg-gray-800 border border-gray-700/60 rounded-lg p-3 space-y-1.5 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className={ev.violation_type === 'helmet_violation' ? 'badge-helmet' : 'badge badge-triple'}>
                      {ev.violation_type === 'helmet_violation' ? 'Helmet Alert' : 'Triple Riding'}
                    </span>
                    <span className="text-xs text-gray-400 font-mono">{((ev.confidence ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                  {ev.vehicle_number && (
                    <div className="flex items-center gap-1.5 py-0.5">
                      <span className="text-[11px] font-mono bg-yellow-950/80 text-yellow-300 border border-yellow-700/70 px-2 py-0.5 rounded font-bold tracking-wider flex items-center gap-1.5 shadow-inner">
                        <span>🇮🇳</span> {ev.vehicle_number}
                      </span>
                    </div>
                  )}
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
                      <div key={rec.filename} className="py-3 text-xs space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2.5">
                            <Film className="w-4 h-4 text-purple-400 shrink-0" />
                            <div>
                              <div className="text-gray-200 font-medium font-mono">{rec.filename}</div>
                              <div className="text-gray-500 text-[11px]">
                                {rec.size_kb} KB • {new Date(rec.created_at).toLocaleString()}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setPreviewRecording(p => p === rec.filename ? null : rec.filename)}
                              className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1"
                            >
                              <Play className="w-3 h-3 text-purple-400" />
                              {previewRecording === rec.filename ? 'Hide' : 'Watch'}
                            </button>
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

                        {previewRecording === rec.filename && (
                          <div className="rounded-xl overflow-hidden border border-purple-800/40 bg-black mt-2 p-1">
                            <video
                              src={`/api/v1/monitoring/recordings/${rec.filename}`}
                              controls
                              autoPlay
                              className="w-full max-h-72 object-contain mx-auto rounded-lg"
                            />
                          </div>
                        )}
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
                    <div key={snap.filename} className="bg-gray-800/60 border border-gray-700/60 rounded-xl overflow-hidden shadow-sm flex flex-col group">
                      <div
                        className="relative cursor-pointer overflow-hidden aspect-video bg-black"
                        onClick={() => setSelectedSnapshot(snap)}
                      >
                        <img
                          src={`/api/v1/monitoring/snapshots/${snap.filename}`}
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = `/evidence/snapshots/${snap.filename}`
                          }}
                          alt={snap.filename}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                        />
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <Eye className="w-5 h-5 text-white" />
                        </div>
                      </div>
                      <div className="p-2 text-[11px] flex-1 flex flex-col justify-between space-y-1">
                        <div className="text-gray-300 font-mono truncate">{snap.filename}</div>
                        <div className="text-gray-500">{snap.size_kb} KB</div>
                        <div className="flex items-center justify-between pt-1 border-t border-gray-700/40">
                          <a
                            href={`/api/v1/monitoring/snapshots/${snap.filename}`}
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

      {/* Snapshot Lightbox / Full-Screen Zoom Modal */}
      {selectedSnapshot && (
        <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl max-w-4xl w-full overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
            <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800 bg-gray-950">
              <div className="flex items-center gap-2">
                <Camera className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-semibold text-white font-mono">{selectedSnapshot.filename}</span>
                <span className="text-xs text-gray-500">({selectedSnapshot.size_kb} KB)</span>
              </div>
              <div className="flex items-center gap-3">
                <a
                  href={`/api/v1/monitoring/snapshots/${selectedSnapshot.filename}`}
                  download
                  className="btn-secondary text-xs py-1 px-3 flex items-center gap-1.5"
                >
                  <Download className="w-3.5 h-3.5 text-green-400" /> Download Image
                </a>
                <button
                  onClick={() => setSelectedSnapshot(null)}
                  className="text-gray-400 hover:text-white p-1 rounded-lg"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
            <div className="p-4 flex-1 flex items-center justify-center overflow-auto bg-black/50">
              <img
                src={`/api/v1/monitoring/snapshots/${selectedSnapshot.filename}`}
                alt={selectedSnapshot.filename}
                className="max-h-[75vh] w-auto max-w-full object-contain rounded-lg shadow-lg"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
