import { useEffect, useState, useCallback } from 'react'
import { Settings, Cpu, Camera, HardDrive, RefreshCw, Save, CheckCircle } from 'lucide-react'
import { getSettings, getModelInfo, configureCamera, updateSettings } from '../services/api'

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
  const [confThresh, setConfThresh] = useState('0.40')
  const [iouThresh, setIouThresh] = useState('0.45')
  const [targetFps, setTargetFps] = useState('10')
  const [helmetFrames, setHelmetFrames] = useState('5')
  const [tripleFrames, setTripleFrames] = useState('5')
  const [dedupSecs, setDedupSecs] = useState('30')
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null)

  const load = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([getSettings(), getModelInfo()])
      setSettings(s)
      setCameraSource(s.camera_source)
      setConfThresh(String(s.confidence_threshold))
      setIouThresh(String(s.iou_threshold))
      setTargetFps(String(s.camera_fps_target))
      setHelmetFrames(String(s.helmet_frames_required))
      setTripleFrames(String(s.triple_frames_required))
      setDedupSecs(String(s.dedup_seconds))
      setModelInfo(m)
    } catch {}
  }, [])

  useEffect(() => { load() }, [load])

  const handleCameraUpdate = async () => {
    try {
      await configureCamera(cameraSource)
      setMessage({ text: 'Camera source updated. Restart monitoring to apply.', ok: true })
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setMessage({ text: err?.response?.data?.detail ?? 'Failed to update camera.', ok: false })
    }
  }

  const handleSaveDynamicSettings = async () => {
    setSaving(true)
    setMessage(null)
    try {
      await updateSettings({
        confidence_threshold: parseFloat(confThresh),
        iou_threshold: parseFloat(iouThresh),
        camera_fps_target: parseInt(targetFps, 10),
        helmet_frames_required: parseInt(helmetFrames, 10),
        triple_frames_required: parseInt(tripleFrames, 10),
        dedup_seconds: parseInt(dedupSecs, 10),
      })
      await load()
      setMessage({ text: '✓ Dynamic settings saved and applied to runtime!', ok: true })
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setMessage({ text: err?.response?.data?.detail ?? 'Failed to save settings.', ok: false })
    } finally {
      setSaving(false)
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
          <h1 className="text-2xl font-bold text-white">System Settings & Thresholds</h1>
          <p className="text-gray-400 text-sm">Configure detection sensitivity, tracking parameters, and camera sources</p>
        </div>
        <button onClick={load} className="btn-secondary flex items-center gap-2 text-sm">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {message && (
        <div className={`rounded-lg px-4 py-3 text-sm flex items-center gap-2 ${message.ok ? 'bg-green-900/30 border border-green-700 text-green-300' : 'bg-red-900/30 border border-red-700 text-red-300'}`}>
          {message.ok && <CheckCircle className="w-4 h-4" />}
          {message.text}
        </div>
      )}

      {/* Dynamic Thresholds Form */}
      <div className="card space-y-4 border border-blue-900/50">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-white flex items-center gap-2">
            <Settings className="w-4 h-4 text-blue-400" />
            Dynamic Detection & Rule Thresholds
          </h2>
          <button
            onClick={handleSaveDynamicSettings}
            disabled={saving}
            className="btn-primary text-xs flex items-center gap-1.5 py-1.5 px-3 disabled:opacity-50"
          >
            <Save className="w-3.5 h-3.5" /> {saving ? 'Saving…' : 'Save Changes'}
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Confidence Threshold ({confThresh})
            </label>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={confThresh}
              onChange={e => setConfThresh(e.target.value)}
              className="w-full accent-blue-500"
            />
            <span className="text-[10px] text-gray-500">Lower for more detections; higher to reduce false alarms</span>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              IoU Threshold ({iouThresh})
            </label>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={iouThresh}
              onChange={e => setIouThresh(e.target.value)}
              className="w-full accent-blue-500"
            />
            <span className="text-[10px] text-gray-500">Non-max suppression bounding-box overlap threshold</span>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Target Capture FPS</label>
            <input
              type="number"
              min="1"
              max="60"
              value={targetFps}
              onChange={e => setTargetFps(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Deduplication Window (Seconds)</label>
            <input
              type="number"
              min="1"
              max="300"
              value={dedupSecs}
              onChange={e => setDedupSecs(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Helmet Violation Required Frames</label>
            <input
              type="number"
              min="1"
              max="30"
              value={helmetFrames}
              onChange={e => setHelmetFrames(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Triple Riding Required Frames</label>
            <input
              type="number"
              min="1"
              max="30"
              value={tripleFrames}
              onChange={e => setTripleFrames(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white"
            />
          </div>
        </div>
      </div>

      {/* Camera */}
      <Section icon={<Camera className="w-4 h-4 text-cyan-400" />} title="Camera Source Configuration">
        <div className="space-y-3">
          <Row label="Active Source" value={settings.camera_source} />
          <Row label="Target Processing Rate" value={`${settings.camera_fps_target} FPS`} />
          <div className="flex gap-2 pt-1">
            <input
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500"
              placeholder='0 = webcam, or path/to/video.webm, rtsp://...'
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
      <Section icon={<Cpu className="w-4 h-4 text-purple-400" />} title="AI Detector Architecture">
        <div className="space-y-2">
          <Row label="Model Weights" value={settings.model_name} />
          <Row label="Weights Status" value={modelInfo?.loaded ? '✓ Model Ready & Loaded' : '✗ Not loaded'} />
          <Row label="Input Resolution" value={`${settings.inference_width} × ${settings.inference_height} px`} />
        </div>
        {modelInfo?.loaded && (
          <div className="mt-3">
            <div className="text-xs text-gray-500 mb-2 font-medium">Supported Classes ({Object.keys(modelInfo.classes).length})</div>
            <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto p-1 bg-gray-900/50 rounded-lg border border-gray-800">
              {Object.entries(modelInfo.classes).map(([id, name]) => (
                <span key={id} className="badge bg-gray-800 text-gray-300 text-[11px] font-mono">{id}: {name}</span>
              ))}
            </div>
            <p className="text-xs text-cyan-400/90 mt-2">
              💡 Helmet analysis uses dedicated head-region HSV color & texture computer vision heuristics on standard COCO models, and supports custom fine-tuned weights when configured.
            </p>
          </div>
        )}
      </Section>

      {/* Storage */}
      <Section icon={<HardDrive className="w-4 h-4 text-green-400" />} title="Evidence Storage & Audit Trail">
        <div className="space-y-2">
          <Row label="Evidence Directory" value={settings.evidence_dir} />
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Evidence crops are stored locally with SHA timestamps and metadata references in SQLite.
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
    <div className="flex items-center justify-between border-b border-gray-800 pb-1 last:border-0 last:pb-0 text-xs">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-200 font-mono">{value}</span>
    </div>
  )
}
