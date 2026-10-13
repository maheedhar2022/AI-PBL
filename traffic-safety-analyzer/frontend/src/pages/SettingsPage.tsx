import { useEffect, useState, useCallback } from 'react'
import { Settings, Cpu, Camera, HardDrive, RefreshCw } from 'lucide-react'
import { getSettings, getModelInfo, configureCamera } from '../services/api'

interface AppSettings {
  model_name: string
  confidence_threshold: number
  iou_threshold: number
  camera_source: string
  camera_fps_target: number
  inference_width: number
  inference_height: number
  evidence_dir: string
  helmet_frames_required: number
  triple_frames_required: number
  dedup_seconds: number
}

interface ModelInfo {
  loaded: boolean
  model_name: string
  classes: Record<string, string>
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [cameraSource, setCameraSource] = useState('')
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null)

  const load = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([getSettings(), getModelInfo()])
      setSettings(s)
      setCameraSource(s.camera_source)
      setModelInfo(m)
    } catch {}
  }, [])

  useEffect(() => { load() }, [load])

  const handleCameraUpdate = async () => {
    try {
      await configureCamera(cameraSource)
      setMessage({ text: 'Camera source updated (restart monitoring to apply).', ok: true })
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setMessage({ text: err?.response?.data?.detail ?? 'Failed to update camera.', ok: false })
    }
  }

  if (!settings) {
    return (
      <div className="p-6 text-gray-500 text-sm">
        Loading settings… Make sure the backend is running.
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 max-w-3xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Settings</h1>
          <p className="text-gray-400 text-sm">Configuration overview (edit via .env to persist)</p>
        </div>
        <button onClick={load} className="btn-secondary flex items-center gap-2 text-sm">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {message && (
        <div className={`rounded-lg px-4 py-3 text-sm ${message.ok ? 'bg-green-900/30 border border-green-700 text-green-300' : 'bg-red-900/30 border border-red-700 text-red-300'}`}>
          {message.text}
        </div>
      )}

      {/* Camera */}
      <Section icon={<Camera className="w-4 h-4 text-blue-400" />} title="Camera Source">
        <div className="space-y-3">
          <Row label="Current Source" value={settings.camera_source} />
          <Row label="Target FPS" value={String(settings.camera_fps_target)} />
          <div className="flex gap-2 pt-1">
            <input
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500"
              placeholder='0 = webcam, or path/to/video.mp4, rtsp://...'
              value={cameraSource}
              onChange={e => setCameraSource(e.target.value)}
            />
            <button onClick={handleCameraUpdate} className="btn-primary text-sm whitespace-nowrap">
              Update Source
            </button>
          </div>
        </div>
      </Section>

      {/* Model */}
      <Section icon={<Cpu className="w-4 h-4 text-purple-400" />} title="AI Model">
        <div className="space-y-2">
          <Row label="Model Name" value={settings.model_name} />
          <Row label="Status" value={modelInfo?.loaded ? '✓ Loaded' : '✗ Not loaded'} />
          <Row label="Confidence Threshold" value={String(settings.confidence_threshold)} />
          <Row label="IoU Threshold" value={String(settings.iou_threshold)} />
          <Row label="Inference Width" value={`${settings.inference_width}px`} />
          <Row label="Inference Height" value={`${settings.inference_height}px`} />
        </div>
        {modelInfo?.loaded && (
          <div className="mt-3">
            <div className="text-xs text-gray-500 mb-2">Supported Classes ({Object.keys(modelInfo.classes).length})</div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(modelInfo.classes).map(([id, name]) => (
                <span key={id} className="badge bg-gray-800 text-gray-300 text-xs">{id}: {name}</span>
              ))}
            </div>
            <p className="text-xs text-yellow-500/80 mt-2">
              ⚠️ Standard COCO models do not include helmet-specific classes. Helmet detection
              uses spatial heuristics or requires a custom fine-tuned model.
            </p>
          </div>
        )}
      </Section>

      {/* Violation Rules */}
      <Section icon={<Settings className="w-4 h-4 text-orange-400" />} title="Violation Rules">
        <div className="space-y-2">
          <Row label="Helmet Frames Required" value={String(settings.helmet_frames_required)} />
          <Row label="Triple Riding Frames Required" value={String(settings.triple_frames_required)} />
          <Row label="Deduplication Window" value={`${settings.dedup_seconds}s`} />
        </div>
        <p className="text-xs text-gray-600 mt-2">
          Edit these values in the <code className="text-gray-400">.env</code> file and restart the backend to apply changes.
        </p>
      </Section>

      {/* Storage */}
      <Section icon={<HardDrive className="w-4 h-4 text-green-400" />} title="Storage">
        <div className="space-y-2">
          <Row label="Evidence Directory" value={settings.evidence_dir} />
        </div>
        <p className="text-xs text-gray-600 mt-2">
          Evidence images are stored locally. Configure <code className="text-gray-400">EVIDENCE_DIR</code> in
          <code className="text-gray-400"> .env</code> to change the location.
        </p>
      </Section>
    </div>
  )
}

function Section({ icon, title, children }: {
  icon: React.ReactNode
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="card space-y-3">
      <h2 className="font-semibold text-white flex items-center gap-2">
        {icon}
        {title}
      </h2>
      {children}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-gray-800 pb-1 last:border-0 last:pb-0">
      <span className="text-gray-500 text-sm">{label}</span>
      <span className="text-gray-200 text-sm font-mono">{value}</span>
    </div>
  )
}

