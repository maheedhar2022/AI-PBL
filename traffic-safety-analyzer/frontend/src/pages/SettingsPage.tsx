import { useEffect, useState, useCallback } from 'react'
import {
  Settings, Cpu, Camera, HardDrive, RefreshCw, Save, CheckCircle,
  Zap, Activity, ShieldAlert, Layers
} from 'lucide-react'
import {
  getSettings, getModelInfo, configureCamera, updateSettings,
  getModelsInfo, runModelsBenchmark
} from '../services/api'

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

interface DeepModelSpec {
  name: string
  type: string
  parameters: number
  input_resolution?: string
  input_features?: number
  embedding_dimension?: number
  classes?: string[]
  device?: string
  loaded?: boolean
}

interface DeepModelsRegistryInfo {
  device: string
  cuda_available: boolean
  torch_version: string
  models: {
    object_detector: DeepModelSpec
    helmet_classifier: DeepModelSpec
    reid_extractor: DeepModelSpec
    violation_classifier: DeepModelSpec
    plate_recognizer?: DeepModelSpec
  }
  total_deep_parameters: number
}

interface BenchmarkResult {
  mean_ms: number
  p95_ms: number
  min_ms: number
  max_ms: number
  throughput_fps: number
}

interface BenchmarkData {
  device: string
  benchmark_iterations: number
  results: {
    yolov8_detection: BenchmarkResult
    helmet_cnn_inference: BenchmarkResult
    reid_embedding_extraction: BenchmarkResult
    violation_mlp_risk: BenchmarkResult
    anpr_plate_recognition?: BenchmarkResult
  }
  timestamp: number
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [dlSpecs, setDlSpecs] = useState<DeepModelsRegistryInfo | null>(null)
  const [benchmarking, setBenchmarking] = useState(false)
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | null>(null)

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
      const [s, m, dl] = await Promise.all([
        getSettings(),
        getModelInfo(),
        getModelsInfo().catch(() => null),
      ])
      setSettings(s)
      setCameraSource(s.camera_source)
      setConfThresh(String(s.confidence_threshold))
      setIouThresh(String(s.iou_threshold))
      setTargetFps(String(s.camera_fps_target))
      setHelmetFrames(String(s.helmet_frames_required))
      setTripleFrames(String(s.triple_frames_required))
      setDedupSecs(String(s.dedup_seconds))
      setModelInfo(m)
      if (dl) setDlSpecs(dl)
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

  const handleRunBenchmark = async () => {
    setBenchmarking(true)
    setMessage(null)
    try {
      const data = await runModelsBenchmark(8)
      setBenchmarkData(data)
      setMessage({ text: '✓ Deep Learning neural benchmark completed on server hardware!', ok: true })
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setMessage({ text: err?.response?.data?.detail ?? 'Benchmark failed.', ok: false })
    } finally {
      setBenchmarking(false)
    }
  }

  const applyPreset = (preset: 'balanced' | 'sensitive' | 'precision' | 'default') => {
    switch (preset) {
      case 'sensitive':
        setConfThresh('0.20')
        setIouThresh('0.35')
        setTargetFps('12')
        setHelmetFrames('3')
        setTripleFrames('3')
        setDedupSecs('20')
        setMessage({ text: 'Applied "High Sensitivity" preset. Click "Save Changes" to apply.', ok: true })
        break
      case 'precision':
        setConfThresh('0.45')
        setIouThresh('0.45')
        setTargetFps('10')
        setHelmetFrames('8')
        setTripleFrames('8')
        setDedupSecs('45')
        setMessage({ text: 'Applied "High Precision" preset. Click "Save Changes" to apply.', ok: true })
        break
      case 'balanced':
      case 'default':
      default:
        setConfThresh('0.25')
        setIouThresh('0.35')
        setTargetFps('10')
        setHelmetFrames('5')
        setTripleFrames('5')
        setDedupSecs('30')
        setMessage({ text: 'Applied "Balanced (Recommended)" defaults. Click "Save Changes" to apply.', ok: true })
        break
    }
  }

  if (!settings) {
    return (
      <div className="p-6 text-gray-500 text-sm flex items-center gap-2">
        <RefreshCw className="w-4 h-4 animate-spin text-blue-400" />
        Loading settings… Make sure the backend is running.
      </div>
    )
  }

  const CAMERA_PRESETS = [
    { label: '🇮🇳 Indian Road Traffic CCTV', path: 'research/datasets/indian_road_traffic_cctv.webm' },
    { label: '🛣️ SF Road Traffic CCTV', path: 'research/datasets/real_cctv_traffic.webm' },
    { label: '🏙️ Urban Roadway CCTV', path: 'research/datasets/motorcycle_traffic_cctv.webm' },
    { label: '🏍️ Motorcycle Test Clip', path: 'research/datasets/test_video.mp4' },
    { label: '💻 System Webcam (0)', path: '0' },
  ]

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">System Settings & Thresholds</h1>
          <p className="text-gray-400 text-sm">Configure AI detection sensitivity, violation rules, and camera input sources</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={load} className="btn-secondary flex items-center gap-1.5 text-xs py-1.5 px-3">
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>
      </div>

      {message && (
        <div className={`rounded-lg px-4 py-3 text-sm flex items-center gap-2 transition-all ${
          message.ok ? 'bg-green-900/30 border border-green-700 text-green-300' : 'bg-red-900/30 border border-red-700 text-red-300'
        }`}>
          {message.ok && <CheckCircle className="w-4 h-4 shrink-0" />}
          <span>{message.text}</span>
        </div>
      )}

      {/* Dynamic Thresholds Form */}
      <div className="card space-y-4 border border-blue-900/50 bg-gray-900/70">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-gray-800 pb-3">
          <div>
            <h2 className="font-semibold text-white flex items-center gap-2">
              <Settings className="w-4 h-4 text-blue-400" />
              Dynamic Detection & Rule Thresholds
            </h2>
            <p className="text-xs text-gray-400 mt-0.5">Parameters apply immediately across the running inference pipeline without restarting the server.</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => applyPreset('default')}
              className="text-xs px-2.5 py-1.5 rounded-lg border border-gray-700 text-gray-300 hover:bg-gray-800 transition-colors"
              title="Reset sliders to recommended factory defaults"
            >
              Reset Defaults
            </button>
            <button
              onClick={handleSaveDynamicSettings}
              disabled={saving}
              className="btn-primary text-xs flex items-center gap-1.5 py-1.5 px-3.5 shadow-sm shadow-blue-500/20 disabled:opacity-50"
            >
              <Save className="w-3.5 h-3.5" /> {saving ? 'Saving…' : 'Save Changes'}
            </button>
          </div>
        </div>

        {/* Quick Sensitivity Presets */}
        <div className="bg-gray-950/60 rounded-lg p-3 border border-gray-800 flex flex-wrap items-center justify-between gap-2">
          <div className="text-xs text-gray-400 font-medium">
            🎯 Quick Sensitivity Presets:
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => applyPreset('balanced')}
              className="text-xs px-2.5 py-1 rounded-md bg-blue-900/40 border border-blue-700/60 text-blue-200 hover:bg-blue-800/60 transition-colors"
            >
              ⚖️ Balanced (Recommended)
            </button>
            <button
              onClick={() => applyPreset('sensitive')}
              className="text-xs px-2.5 py-1 rounded-md bg-amber-900/40 border border-amber-700/60 text-amber-200 hover:bg-amber-800/60 transition-colors"
            >
              🔍 High Sensitivity (Catch all)
            </button>
            <button
              onClick={() => applyPreset('precision')}
              className="text-xs px-2.5 py-1 rounded-md bg-purple-900/40 border border-purple-700/60 text-purple-200 hover:bg-purple-800/60 transition-colors"
            >
              🎯 High Precision (Strict)
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 pt-1">
          {/* Confidence Slider */}
          <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-800">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-gray-300">Confidence Threshold</label>
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-800/60">{confThresh}</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={confThresh}
              onChange={e => setConfThresh(e.target.value)}
              className="w-full accent-blue-500 cursor-pointer"
            />
            <p className="text-[11px] text-gray-400 mt-1">
              Minimum AI probability required to register detections. Lower values detect smaller objects; higher values eliminate false alarms.
            </p>
          </div>

          {/* IoU Slider */}
          <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-800">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-gray-300">IoU Overlap Threshold</label>
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-800/60">{iouThresh}</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={iouThresh}
              onChange={e => setIouThresh(e.target.value)}
              className="w-full accent-blue-500 cursor-pointer"
            />
            <p className="text-[11px] text-gray-400 mt-1">
              Non-Max Suppression bounding box overlap threshold. Prevents duplicate detections on the same vehicle.
            </p>
          </div>

          {/* Target Capture FPS */}
          <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-800">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-gray-300">Target Pipeline FPS</label>
              <span className="font-mono text-xs text-gray-400">{targetFps} FPS</span>
            </div>
            <input
              type="number"
              min="1"
              max="60"
              value={targetFps}
              onChange={e => setTargetFps(e.target.value)}
              className="w-full bg-gray-850 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
            <p className="text-[11px] text-gray-400 mt-1">
              Sampling frequency for video frames. 10–15 FPS provides optimal real-time tracking accuracy on CPU/GPU.
            </p>
          </div>

          {/* Deduplication Window */}
          <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-800">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-gray-300">Deduplication Window</label>
              <span className="font-mono text-xs text-gray-400">{dedupSecs} seconds</span>
            </div>
            <input
              type="number"
              min="1"
              max="300"
              value={dedupSecs}
              onChange={e => setDedupSecs(e.target.value)}
              className="w-full bg-gray-850 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
            <p className="text-[11px] text-gray-400 mt-1">
              Minimum duration before raising another violation for the same tracked vehicle ID to avoid duplicate alerts.
            </p>
          </div>

          {/* Helmet Violation Frames */}
          <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-800">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-gray-300">Helmet Violation Persistence</label>
              <span className="font-mono text-xs text-gray-400">{helmetFrames} frames</span>
            </div>
            <input
              type="number"
              min="1"
              max="30"
              value={helmetFrames}
              onChange={e => setHelmetFrames(e.target.value)}
              className="w-full bg-gray-850 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
            <p className="text-[11px] text-gray-400 mt-1">
              Number of consecutive video frames the helmet violation must persist before logging an incident (filters transient occlusion).
            </p>
          </div>

          {/* Triple Riding Frames */}
          <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-800">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-gray-300">Triple Riding Persistence</label>
              <span className="font-mono text-xs text-gray-400">{tripleFrames} frames</span>
            </div>
            <input
              type="number"
              min="1"
              max="30"
              value={tripleFrames}
              onChange={e => setTripleFrames(e.target.value)}
              className="w-full bg-gray-850 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
            <p className="text-[11px] text-gray-400 mt-1">
              Number of consecutive frames with ≥3 riders detected before recording a triple riding safety violation.
            </p>
          </div>
        </div>
      </div>

      {/* Camera */}
      <Section icon={<Camera className="w-4 h-4 text-cyan-400" />} title="Camera Source Configuration">
        <div className="space-y-3">
          <Row label="Active Source" value={settings.camera_source} />
          <Row label="Target Processing Rate" value={`${settings.camera_fps_target} FPS`} />

          {/* Quick presets */}
          <div>
            <div className="text-xs text-gray-400 mb-1.5 font-medium">Quick Select Feed:</div>
            <div className="flex flex-wrap gap-2">
              {CAMERA_PRESETS.map(p => (
                <button
                  key={p.path}
                  type="button"
                  onClick={() => setCameraSource(p.path)}
                  className={`text-xs px-2.5 py-1 rounded-md border transition-colors ${
                    cameraSource === p.path
                      ? 'bg-cyan-950 border-cyan-600 text-cyan-200'
                      : 'bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-750'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex gap-2 pt-1">
            <input
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
              placeholder='0 = webcam, or path/to/video.webm, rtsp://...'
              value={cameraSource}
              onChange={e => setCameraSource(e.target.value)}
            />
            <button onClick={handleCameraUpdate} className="btn-primary text-sm whitespace-nowrap">
              Update Source
            </button>
          </div>
          <p className="text-[11px] text-gray-400">
            Supports local USB/integrated webcams (<code className="text-gray-300 font-mono">0</code>), open-source video files, and remote CCTV streams (<code className="text-gray-300 font-mono">rtsp://</code>, <code className="text-gray-300 font-mono">http://</code>).
          </p>
        </div>
      </Section>

      {/* Deep Learning Models Registry & Benchmarking */}
      <div className="card space-y-4 border border-purple-900/50 bg-gray-900/70">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-800 pb-3">
          <div>
            <h2 className="font-semibold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-purple-400" />
              Deep Learning Neural Networks & Model Zoo
            </h2>
            <p className="text-xs text-gray-400 mt-0.5">
              PyTorch deep neural network models running locally for multi-stage vehicle, helmet, Re-ID, and risk inference.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] px-2.5 py-1 rounded bg-purple-950/80 border border-purple-800 text-purple-300 font-mono">
              Device: {dlSpecs?.device.toUpperCase() ?? 'CPU'} (PyTorch {dlSpecs?.torch_version?.split('+')[0] ?? '2.14'})
            </span>
            <button
              onClick={handleRunBenchmark}
              disabled={benchmarking}
              className="btn-secondary text-xs flex items-center gap-1.5 py-1.5 px-3 border-purple-700/60 text-purple-200 hover:bg-purple-900/40 disabled:opacity-50"
            >
              <Zap className={`w-3.5 h-3.5 ${benchmarking ? 'animate-spin text-yellow-400' : 'text-purple-400'}`} />
              {benchmarking ? 'Benchmarking Neural Nets…' : '⚡ Run DL Benchmark'}
            </button>
          </div>
        </div>

        {/* Deep Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
          {/* 1. YOLOv8 Object Detector */}
          <div className="bg-gray-800/40 p-3.5 rounded-lg border border-gray-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-blue-400" />
                YOLOv8 Deep Object Detector
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-950 border border-blue-800 text-blue-300 font-mono">
                3.15M params
              </span>
            </div>
            <p className="text-[11px] text-gray-400">
              CSPDarknet53 backbone + PANet feature pyramid for multiclass vehicle & pedestrian detection.
            </p>
            <div className="text-[10px] font-mono text-gray-400 space-y-0.5 pt-1 border-t border-gray-800/60">
              <div>Input: 3×640×640 px · Status: {modelInfo?.loaded ? '✓ Loaded' : '✗ Offline'}</div>
              <div>Classes: 80 COCO categories (Car, Motorcycle, Bus, Truck, Person)</div>
            </div>
          </div>

          {/* 2. DeepHelmetCNN-SE */}
          <div className="bg-gray-800/40 p-3.5 rounded-lg border border-gray-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                DeepHelmetCNN + SE Attention
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-950 border border-amber-800 text-amber-300 font-mono">
                {dlSpecs?.models?.helmet_classifier?.parameters ? `${Math.round(dlSpecs.models.helmet_classifier.parameters / 1000)}k params` : '184k params'}
              </span>
            </div>
            <p className="text-[11px] text-gray-400">
              3-Stage Residual CNN with Squeeze-and-Excitation channel attention for head-crop compliance classification.
            </p>
            <div className="text-[10px] font-mono text-gray-400 space-y-0.5 pt-1 border-t border-gray-800/60">
              <div>Input: 3×64×64 RGB Head Crop · Device: {dlSpecs?.device ?? 'cpu'}</div>
              <div>Outputs: [helmet, no_helmet, uncertain] Softmax Logits</div>
            </div>
          </div>

          {/* 3. DeepReIDNet */}
          <div className="bg-gray-800/40 p-3.5 rounded-lg border border-gray-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                DeepReIDNet Appearance Embedding
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 border border-cyan-800 text-cyan-300 font-mono">
                {dlSpecs?.models?.reid_extractor?.parameters ? `${Math.round(dlSpecs.models.reid_extractor.parameters / 1000)}k params` : '480k params'}
              </span>
            </div>
            <p className="text-[11px] text-gray-400">
              Projects vehicle & rider crops onto a 128-d unit hypersphere for cross-frame Re-ID in Deep SORT tracking.
            </p>
            <div className="text-[10px] font-mono text-gray-400 space-y-0.5 pt-1 border-t border-gray-800/60">
              <div>Embedding: 128-dim L2-normalized vector · Metric: Cosine Distance</div>
              <div>Resolution: 3×128×64 px Bounding Box Patches</div>
            </div>
          </div>

          {/* 4. ViolationRiskMLP */}
          <div className="bg-gray-800/40 p-3.5 rounded-lg border border-gray-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-green-400" />
                ViolationRiskMLP Spatio-Temporal
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-950 border border-green-800 text-green-300 font-mono">
                {dlSpecs?.models?.violation_classifier?.parameters ? `${dlSpecs.models.violation_classifier.parameters} params` : '7.7k params'}
              </span>
            </div>
            <p className="text-[11px] text-gray-400">
              Deep Multi-Layer Perceptron with kinematic and geometric representations for multi-rider risk scoring.
            </p>
            <div className="text-[10px] font-mono text-gray-400 space-y-0.5 pt-1 border-t border-gray-800/60">
              <div>Input: 16-D Spatio-Temporal State (Speed, Dispersion, IoU, Accel)</div>
              <div>Classes: [Safe, Triple Riding, Dangerous Maneuver]</div>
            </div>
          </div>

          {/* 5. IndianANPR-DeepPlateOCRNet */}
          <div className="bg-gray-800/40 p-3.5 rounded-lg border border-gray-800 space-y-2 md:col-span-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                <span className="text-sm">🇮🇳</span>
                IndianANPR-DeepPlateOCRNet (MoRTH HSRP)
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-950 border border-indigo-800 text-indigo-300 font-mono">
                {dlSpecs?.models?.plate_recognizer?.parameters ? `${Math.round(dlSpecs.models.plate_recognizer.parameters / 1000)}k params` : '71k params'}
              </span>
            </div>
            <p className="text-[11px] text-gray-400">
              Residual Deep CNN OCR engine with morphological Sobel ROI localization and grammar validation for Indian High Security Registration Plates across all 28+ states and UTs.
            </p>
            <div className="text-[10px] font-mono text-gray-400 space-y-0.5 pt-1 border-t border-gray-800/60">
              <div>Input: 1×32×96 Grayscale Plate ROI · Format: [State 2L] [RTO 2D] [Series 1-2L] [Number 4D]</div>
              <div>Grammar: Indian MoRTH Syntax Sanitizer & Character Substitution Disambiguation (O/0, B/8, I/1)</div>
            </div>
          </div>
        </div>

        {/* Live Benchmark Results Display */}
        {benchmarkData && (
          <div className="bg-purple-950/30 border border-purple-800/60 rounded-lg p-3 space-y-2 animate-fadeIn">
            <div className="flex items-center justify-between text-xs text-purple-200 font-medium">
              <span>⚡ Live Neural Network Hardware Benchmark ({benchmarkData.benchmark_iterations} iterations on {benchmarkData.device.toUpperCase()})</span>
              <span className="font-mono text-[10px] text-purple-300">Total Params: ~3.90M</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-1 text-center">
              <div className="bg-gray-900/80 p-2 rounded border border-gray-800">
                <div className="text-[10px] text-gray-400">YOLOv8 Detection</div>
                <div className="text-sm font-bold text-white font-mono">{benchmarkData.results.yolov8_detection.mean_ms} ms</div>
                <div className="text-[10px] text-blue-400 font-mono">{benchmarkData.results.yolov8_detection.throughput_fps} FPS</div>
              </div>
              <div className="bg-gray-900/80 p-2 rounded border border-gray-800">
                <div className="text-[10px] text-gray-400">Helmet CNN (SE)</div>
                <div className="text-sm font-bold text-white font-mono">{benchmarkData.results.helmet_cnn_inference.mean_ms} ms</div>
                <div className="text-[10px] text-amber-400 font-mono">{benchmarkData.results.helmet_cnn_inference.throughput_fps} FPS</div>
              </div>
              <div className="bg-gray-900/80 p-2 rounded border border-gray-800">
                <div className="text-[10px] text-gray-400">Re-ID Embedding</div>
                <div className="text-sm font-bold text-white font-mono">{benchmarkData.results.reid_embedding_extraction.mean_ms} ms</div>
                <div className="text-[10px] text-cyan-400 font-mono">{benchmarkData.results.reid_embedding_extraction.throughput_fps} FPS</div>
              </div>
              <div className="bg-gray-900/80 p-2 rounded border border-gray-800">
                <div className="text-[10px] text-gray-400">Violation Risk MLP</div>
                <div className="text-sm font-bold text-white font-mono">{benchmarkData.results.violation_mlp_risk.mean_ms} ms</div>
                <div className="text-[10px] text-green-400 font-mono">{benchmarkData.results.violation_mlp_risk.throughput_fps} FPS</div>
              </div>
              {benchmarkData.results.anpr_plate_recognition && (
                <div className="bg-gray-900/80 p-2 rounded border border-gray-800">
                  <div className="text-[10px] text-gray-400">Indian ANPR OCR</div>
                  <div className="text-sm font-bold text-white font-mono">{benchmarkData.results.anpr_plate_recognition.mean_ms} ms</div>
                  <div className="text-[10px] text-yellow-400 font-mono">{benchmarkData.results.anpr_plate_recognition.throughput_fps} FPS</div>
                </div>
              )}
            </div>
          </div>
        )}

        {modelInfo?.loaded && (
          <div className="pt-2 border-t border-gray-800">
            <div className="text-xs text-gray-400 mb-2 font-medium">Detector Supported Classes ({Object.keys(modelInfo.classes).length})</div>
            <div className="flex flex-wrap gap-1.5 max-h-28 overflow-y-auto p-1 bg-gray-950/60 rounded-lg border border-gray-800">
              {Object.entries(modelInfo.classes).map(([id, name]) => (
                <span key={id} className="badge bg-gray-800 text-gray-300 text-[11px] font-mono">{id}: {name}</span>
              ))}
            </div>
          </div>
        )}
      </div>

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
