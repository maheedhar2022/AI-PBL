import { useEffect, useState, useCallback, useRef } from 'react'
import {
  Play, Square, Camera, Video, AlertTriangle, Car, Bike, Bus, Truck, Users, Activity
} from 'lucide-react'
import {
  startMonitoring, stopMonitoring, getMonitoringStatus, getLatestFrame, configureCamera
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
  const frameTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getMonitoringStatus()
      setRunning(status.pipeline_running)
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
    await stopMonitoring()
    setRunning(false)
    setFrame(null)
  }

  const counts = metrics.vehicle_counts || {}

  return (
    <div className="p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Live CCTV Monitoring & Traffic Analytics</h1>
          <p className="text-gray-400 text-sm">Real-time AI vehicle detection, tracking, and road safety analysis</p>
        </div>
        <div className="flex items-center gap-2">
          {running
            ? <span className="flex items-center gap-1.5 text-green-400 text-sm font-medium"><span className="w-2.5 h-2.5 bg-green-400 rounded-full animate-pulse" />Live CCTV Stream</span>
            : <span className="flex items-center gap-1.5 text-gray-500 text-sm"><span className="w-2.5 h-2.5 bg-gray-600 rounded-full" />Feed Offline</span>
          }
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-300 text-sm">{error}</div>
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
            {!running
              ? <button onClick={handleStart} className="btn-primary flex items-center gap-2">
                  <Play className="w-4 h-4" /> Start Feed
                </button>
              : <button onClick={handleStop} className="btn-danger flex items-center gap-2">
                  <Square className="w-4 h-4" /> Stop Feed
                </button>
            }
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
                <span className="text-xs text-gray-700 max-w-sm text-center">Open-Source CCTV feed loops continuously with full vehicle detection and tracking.</span>
              </div>
            )}
            {running && (
              <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-red-600/90 rounded-md px-2 py-0.5 text-xs text-white shadow">
                <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                CCTV ON AIR
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
                    <span className="text-xs text-gray-400 font-mono">{(ev.confidence ?? 0 * 100).toFixed(0)}%</span>
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
    </div>
  )
}
