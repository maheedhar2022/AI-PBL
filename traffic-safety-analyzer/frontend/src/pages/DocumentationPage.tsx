import { BookOpen, AlertTriangle, Info, Cpu, Database, Camera } from 'lucide-react'

const sections = [
  {
    icon: <Info className="w-5 h-5 text-blue-400" />,
    title: 'Project Overview',
    content: `AI-Based Traffic Violation & Road Safety Analyzer is a research-oriented computer vision 
    application that processes live camera footage to detect potential traffic violations.
    
    It detects motorcycles and their riders, identifies potential helmet violations and triple-riding 
    incidents, and presents results through a real-time monitoring dashboard. All detections are 
    presented as potential violations that require human review — the system does not autonomously 
    confirm legal violations.`,
  },
  {
    icon: <AlertTriangle className="w-5 h-5 text-orange-400" />,
    title: 'Supported Violation Types',
    content: null,
    custom: (
      <div className="space-y-4">
        <div className="bg-gray-800 rounded-lg p-4 space-y-2">
          <div className="font-medium text-orange-300">1. Helmet Violation</div>
          <p className="text-sm text-gray-400">
            Detects motorcyclists who may not be wearing helmets. Requires a custom helmet-trained
            model for reliable detection. With standard COCO models, the system uses spatial
            heuristics and marks results as <code className="text-yellow-400">uncertain</code>.
          </p>
          <div className="text-xs text-gray-600">
            Outcome labels: <code>potential_violation</code> | <code>no_violation_detected</code> | <code>uncertain</code>
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4 space-y-2">
          <div className="font-medium text-purple-300">2. Triple Riding</div>
          <p className="text-sm text-gray-400">
            Detects motorcycles carrying more than two riders. The system counts associated
            persons across consecutive frames and flags events when more than 2 riders are
            consistently detected on the same motorcycle.
          </p>
          <div className="text-xs text-gray-600">
            Threshold: configurable <code>TRIPLE_FRAMES_REQUIRED</code> frames (default: 5)
          </div>
        </div>
      </div>
    ),
  },
  {
    icon: <Cpu className="w-5 h-5 text-purple-400" />,
    title: 'AI Model & Limitations',
    content: null,
    custom: (
      <div className="space-y-3 text-sm text-gray-400">
        <p>
          The system uses a pretrained YOLOv8 model (default: <code className="text-gray-300">yolov8n.pt</code>)
          trained on the COCO dataset. COCO provides classes for <code className="text-gray-300">person</code>,
          <code className="text-gray-300"> motorcycle</code>, <code className="text-gray-300">bicycle</code>, etc.
        </p>
        <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-3">
          <div className="font-medium text-yellow-400 mb-1">⚠️ Known Limitations</div>
          <ul className="list-disc list-inside space-y-1 text-xs">
            <li>Standard COCO models have NO helmet/no-helmet classes — helmet detection requires fine-tuning.</li>
            <li>Rider-to-motorcycle association uses spatial IoU heuristics — may fail with dense traffic.</li>
            <li>Performance degrades in low-light, heavy rain, or very small image sizes.</li>
            <li>The system is conservative: uncertain detections are not flagged as violations.</li>
            <li>CPU inference is slow (~1–3 FPS); GPU is strongly recommended for real-time use.</li>
          </ul>
        </div>
        <p>
          To improve helmet detection, train or fine-tune a YOLO model on a helmet dataset
          (e.g., the Safety Helmet Detection dataset on Roboflow) and configure
          <code className="text-gray-300"> HELMET_PRESENT_CLASSES</code> and
          <code className="text-gray-300"> HELMET_ABSENT_CLASSES</code> in
          <code className="text-gray-300"> backend/config/constants.py</code>.
        </p>
      </div>
    ),
  },
  {
    icon: <Database className="w-5 h-5 text-green-400" />,
    title: 'Dataset & Evaluation Methodology',
    content: null,
    custom: (
      <div className="space-y-3 text-sm text-gray-400">
        <p>
          The evaluation pipeline is an offline, reproducible system that accepts any YOLO-format
          labeled dataset. It computes standard metrics: Precision, Recall, F1-score, mAP@50, and
          mAP@50–95.
        </p>
        <div className="bg-gray-800 rounded-lg p-3 space-y-2">
          <div className="font-medium text-gray-200">Expected Dataset Layout (YOLO format)</div>
          <pre className="text-xs text-green-300 font-mono">{`research/datasets/my_dataset/
├── images/
│   └── val/         ← validation images (.jpg/.png)
└── labels/
    └── val/         ← YOLO annotation .txt files`}</pre>
          <div className="text-xs text-gray-500">
            Each label file: <code>class_id cx cy w h</code> (normalised 0–1, one detection per line)
          </div>
        </div>
        <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-3 text-xs">
          <div className="font-medium text-blue-400 mb-1">⚠️ Data Leakage Prevention</div>
          Split data by recording session or camera, not by randomly sampling adjacent frames.
          The evaluation pipeline never touches training data.
        </div>
        <p>
          All evaluation results show <strong className="text-white">Not evaluated</strong> until
          real labeled data is provided and an evaluation run is completed. No benchmark numbers
          are fabricated.
        </p>
      </div>
    ),
  },
  {
    icon: <Camera className="w-5 h-5 text-blue-400" />,
    title: 'Operating the System',
    content: null,
    custom: (
      <ol className="list-decimal list-inside space-y-2 text-sm text-gray-400">
        <li>Start the backend: <code className="text-gray-300">cd backend && uvicorn main:app --reload</code></li>
        <li>Start the frontend: <code className="text-gray-300">cd frontend && npm run dev</code></li>
        <li>Open <code className="text-gray-300">http://localhost:5173</code> in your browser.</li>
        <li>Go to <b className="text-white">Settings</b> to verify camera source (default: webcam index 0). For a test video, set the camera source to the full path of a video file.</li>
        <li>Go to <b className="text-white">Live Monitoring</b> and click <b className="text-white">Start</b>.</li>
        <li>Detections will appear in real time. Potential violations generate incident records.</li>
        <li>Go to <b className="text-white">Incidents</b> to review, confirm, or dismiss alerts.</li>
        <li>To evaluate model performance, add a YOLO-format labeled dataset to <code className="text-gray-300">research/datasets/</code> and run an evaluation from the <b className="text-white">AI Evaluation</b> page.</li>
      </ol>
    ),
  },
]

export default function DocumentationPage() {
  return (
    <div className="p-6 space-y-6 max-w-3xl">
      <div className="flex items-center gap-3">
        <BookOpen className="w-7 h-7 text-blue-400" />
        <div>
          <h1 className="text-2xl font-bold text-white">Documentation</h1>
          <p className="text-gray-400 text-sm">System guide, model limitations, and dataset instructions</p>
        </div>
      </div>

      {sections.map((sec, i) => (
        <div key={i} className="card space-y-3">
          <h2 className="flex items-center gap-2 font-semibold text-white text-lg">
            {sec.icon}
            {sec.title}
          </h2>
          {sec.content && (
            <p className="text-gray-400 text-sm whitespace-pre-line">{sec.content}</p>
          )}
          {sec.custom}
        </div>
      ))}

      {/* Architecture Diagram */}
      <div className="card space-y-3">
        <h2 className="font-semibold text-white text-lg">Processing Pipeline</h2>
        <div className="bg-gray-800 rounded-lg p-4 font-mono text-xs text-green-300 space-y-1">
          {[
            'Camera / Video File',
            '  ↓  OpenCV Capture (thread)',
            '  ↓  Frame Buffer (bounded queue)',
            '  ↓  AI Object Detection (YOLO)',
            '  ↓  Multi-Object Tracking (IoU tracker)',
            '  ↓  Rider → Motorcycle Association',
            '  ↓  Helmet Violation Rules (temporal)',
            '  ↓  Triple Riding Rules (temporal)',
            '  ↓  Deduplication (per-track window)',
            '  ↓  Evidence Image Capture',
            '  ↓  SQLite Incident Storage',
            '  ↓  WebSocket → Dashboard',
          ].map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      </div>
    </div>
  )
}

